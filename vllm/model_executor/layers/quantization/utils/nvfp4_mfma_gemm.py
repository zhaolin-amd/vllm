# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 GEMM on CDNA4 (gfx950) via the FP4 MFMA engine.

Strategy (see docs/design/nvfp4_on_rocm_mfma.md):

CDNA4's scaled-MFMA (`tl.dot_scaled(..., "e2m1", ...)`) hard-wires the
microscale to E8M0 over blocks of 32 -- that is exactly OCP MXFP4 and is
*not* what NVFP4 uses (E4M3 over blocks of 16, plus a per-tensor fp32
global scale). We borrow only the FP4x4 multiply-accumulate: feeding the
E8M0 scale a neutral 127 (== 2^0 == 1) turns the hardware scaling into an
identity, yielding an *unscaled* fp4xfp4 -> fp32 partial product. NVFP4's
real scales are then applied in fp32 software:

    out[m,n] = alpha * sum_blocks {
                   (a_sf16_b * w_sf16_b) * sum_within_16 e2m1_a * e2m1_w }

The block (E4M3) scale must be applied inside the K loop on the block-16
boundary -- once the dot sums across a block boundary the per-block
contributions can no longer be separated. The global scale is a
per-tensor constant (``alpha``) folded into the epilogue.

This module is import-safe without triton; callers must gate on
``nvfp4_mfma_gemm_available()``.
"""

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX_RECIPROCAL,
    get_reciprocal,
)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# NVFP4 block (group) size along K.
NVFP4_BLOCK_SIZE = 16
# E8M0 byte that decodes to 2^0 == 1.0; feeding this to dot_scaled makes the
# hardware microscaling an identity so we recover the raw fp4xfp4 product.
NEUTRAL_E8M0 = 127


def nvfp4_mfma_gemm_available() -> bool:
    return _HAS_TRITON


if _HAS_TRITON:

    @triton.jit
    def _nvfp4_mfma_gemm_kernel(
        a_ptr,  # (M, K//2) uint8, packed E2M1 activations
        b_ptr,  # (N, K//2) uint8, packed E2M1 weights
        c_ptr,  # (M, N) output
        a_sf_ptr,  # (M, K//16) float8_e4m3fn activation block scales
        b_sf_ptr,  # (N, K//16) float8_e4m3fn weight block scales
        alpha,  # fp32 scalar: a_global * w_global
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  # in *elements*; must be a multiple of 16
    ):
        """C = (A @ B^T) with NVFP4 operands, scales applied in fp32.

        A and B are E2M1 packed two-per-byte. ``BLOCK_SIZE_K`` counts FP4
        elements; the byte dimension is ``BLOCK_SIZE_K // 2``. Each block of
        ``NVFP4_BLOCK_SIZE`` (16) elements along K shares one E4M3 scale.
        """
        # Literal 16: triton forbids reading module globals (NVFP4_BLOCK_SIZE)
        # inside @jit unless they are tl.constexpr.
        SCALE_GROUP_SIZE: tl.constexpr = 16
        SCALES_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // SCALE_GROUP_SIZE

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        # Byte offsets along K (two FP4 values packed per uint8).
        offs_k_byte = tl.arange(0, BLOCK_SIZE_K // 2)

        a_ptrs = a_ptr + (
            offs_m[:, None] * stride_am + offs_k_byte[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_n[:, None] * stride_bn + offs_k_byte[None, :] * stride_bk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Literal 127 (== NEUTRAL_E8M0): see SCALE_GROUP_SIZE note above.
        neutral = tl.full(
            (BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE), 127, tl.uint8
        )
        neutral_b = tl.full(
            (BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE), 127, tl.uint8
        )

        num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        for kt in range(num_k_tiles):
            k_base = kt * BLOCK_SIZE_K
            byte_base = kt * (BLOCK_SIZE_K // 2)
            k_mask = offs_k_byte[None, :] < (K // 2 - byte_base)

            a = tl.load(a_ptrs, mask=k_mask, other=0)
            b = tl.load(b_ptrs, mask=k_mask, other=0)

            # Apply each block-16 scale separately: an unscaled fp4 dot over
            # the 16 elements of the block, then a fp32 outer-product of the
            # E4M3 row/col scales. Summing block contributions in fp32 is
            # exactly NVFP4 semantics.
            for sb in tl.static_range(SCALES_PER_BLOCK):
                lo = sb * (SCALE_GROUP_SIZE // 2)
                a_blk = tl.where(
                    (offs_k_byte[None, :] >= lo)
                    & (offs_k_byte[None, :] < lo + SCALE_GROUP_SIZE // 2),
                    a,
                    0,
                )
                b_blk = tl.where(
                    (offs_k_byte[None, :] >= lo)
                    & (offs_k_byte[None, :] < lo + SCALE_GROUP_SIZE // 2),
                    b,
                    0,
                )
                # Unscaled fp4xfp4 -> fp32 (E8M0 fed neutral 127 == x1).
                p = tl.dot_scaled(
                    a_blk, neutral, "e2m1", b_blk.T, neutral_b.T, "e2m1"
                )

                sk = k_base // SCALE_GROUP_SIZE + sb
                a_sf = tl.load(
                    a_sf_ptr + offs_m * stride_asm + sk * stride_ask,
                    mask=offs_m < M,
                    other=0.0,
                ).to(tl.float32)
                b_sf = tl.load(
                    b_sf_ptr + offs_n * stride_bsn + sk * stride_bsk,
                    mask=offs_n < N,
                    other=0.0,
                ).to(tl.float32)
                accumulator += p * (a_sf[:, None] * b_sf[None, :])

            a_ptrs += (BLOCK_SIZE_K // 2) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

        accumulator *= alpha
        c = accumulator.to(c_ptr.type.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def quantize_nvfp4_activation(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    block_size: int = NVFP4_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP16 activation to NVFP4 (packed E2M1 + E4M3 scales).

    Mirrors the reference quantizer (``ref_nvfp4_quant``) but returns the
    packed uint8 tensor and E4M3 block scales the MFMA kernel consumes.

    Args:
        x: Activation, shape ``(M, K)``.
        input_global_scale: Per-tensor fp32 scalar (``1 / input_global``).
        block_size: K group size, must be 16 for NVFP4.

    Returns:
        ``(x_fp4, x_sf)`` with ``x_fp4`` shape ``(M, K // 2)`` uint8 and
        ``x_sf`` shape ``(M, K // block_size)`` float8_e4m3fn.
    """
    assert x.ndim == 2
    m, k = x.shape
    assert k % block_size == 0
    xb = x.reshape(m, k // block_size, block_size).to(torch.float32)
    vec_max = torch.max(torch.abs(xb), dim=-1, keepdim=True)[0]
    scale = input_global_scale * (vec_max * FLOAT4_E2M1_MAX_RECIPROCAL)
    scale = torch.clamp(scale, min=-448, max=448).to(torch.float8_e4m3fn)
    scale_f32 = scale.to(torch.float32)
    out_scale = get_reciprocal(scale_f32 * get_reciprocal(input_global_scale))
    scaled = torch.clamp(xb * out_scale, -6.0, 6.0).reshape(m, k)
    x_fp4 = _pack_e2m1(scaled)
    return x_fp4, scale.reshape(m, k // block_size)


def _pack_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values in [-6, 6] to E2M1 and pack two per uint8.

    Magnitude thresholds match ``cast_to_fp4`` / ``_round_to_fp4``; the
    E2M1 magnitude codes map to {0, .5, 1, 1.5, 2, 3, 4, 6}.
    """
    sign = (x < 0).to(torch.uint8)
    ax = torch.abs(x)
    # Boundaries between adjacent E2M1 magnitudes, half-open on the side that
    # matches the reference rounding.
    mag = torch.zeros_like(ax, dtype=torch.uint8)
    for code, lo, inclusive in (
        (1, 0.25, False),
        (2, 0.75, True),
        (3, 1.25, False),
        (4, 1.75, True),
        (5, 2.5, False),
        (6, 3.5, True),
        (7, 5.0, False),
    ):
        cond = ax >= lo if inclusive else ax > lo
        mag = torch.where(cond, torch.full_like(mag, code), mag)
    nib = (sign << 3) | mag
    nib = nib.reshape(*nib.shape[:-1], -1, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).to(torch.uint8)


def gemm_nvfp4_mfma(
    x_fp4: torch.Tensor,
    w_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    w_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Y = (A @ B^T) for NVFP4 operands using the FP4 MFMA engine.

    Args:
        x_fp4: Packed E2M1 activations, ``(M, K // 2)`` uint8.
        w_fp4: Packed E2M1 weights, ``(N, K // 2)`` uint8.
        a_sf: Activation E4M3 block scales, ``(M, K // 16)``.
        w_sf: Weight E4M3 block scales, ``(N, K // 16)``.
        alpha: Per-tensor fp32 scalar ``a_global * w_global``.
        out_dtype: Output dtype (BF16/FP16).

    Returns:
        ``(M, N)`` output tensor.
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is required for gemm_nvfp4_mfma")

    m = x_fp4.shape[0]
    n = w_fp4.shape[0]
    k = x_fp4.shape[1] * 2
    assert w_fp4.shape[1] * 2 == k
    assert k % NVFP4_BLOCK_SIZE == 0

    a_sf = a_sf.view(torch.float8_e4m3fn)
    w_sf = w_sf.view(torch.float8_e4m3fn)
    # Pass a python float, not a 0-d tensor: triton would bind the latter as a
    # pointer<fp32>, breaking the scalar ``accumulator *= alpha`` in the kernel.
    alpha_f = float(alpha)

    y = torch.empty(m, n, dtype=out_dtype, device=x_fp4.device)

    block_m, block_n = 64, 64
    # K tile must be a multiple of 16 and span enough work per MFMA.
    block_k = min(128, k)
    block_k = (block_k // NVFP4_BLOCK_SIZE) * NVFP4_BLOCK_SIZE

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _nvfp4_mfma_gemm_kernel[grid](
        x_fp4,
        w_fp4,
        y,
        a_sf,
        w_sf,
        alpha_f,
        m,
        n,
        k,
        x_fp4.stride(0),
        x_fp4.stride(1),
        w_fp4.stride(0),
        w_fp4.stride(1),
        y.stride(0),
        y.stride(1),
        a_sf.stride(0),
        a_sf.stride(1),
        w_sf.stride(0),
        w_sf.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )
    return y
