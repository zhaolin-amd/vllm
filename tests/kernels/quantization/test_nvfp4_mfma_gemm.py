# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical check for the gfx950 NVFP4 FP4-MFMA GEMM (scheme B1).

Compares ``gemm_nvfp4_mfma`` against the emulation reference
(``run_nvfp4_emulations``). Both should land at the same NVFP4 numerical
level; we allow a tolerance for fp4 quantization noise and fp32 accumulation
order differences. Skipped off gfx950.
"""

import pytest
import torch

from vllm.platforms import current_platform


def _is_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_gfx950
    except ImportError:
        return False
    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _is_gfx950(),
    reason="NVFP4 FP4-MFMA kernel requires gfx950 (CDNA4).",
)


@pytest.mark.parametrize(
    "m, n, k",
    [
        (16, 64, 64),
        (32, 128, 256),
        (64, 256, 512),
        (128, 512, 4096),
    ],
)
def test_nvfp4_mfma_matches_emulation(m: int, n: int, k: int) -> None:
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        FLOAT4_E2M1_MAX_RECIPROCAL,
        dequantize_to_dtype,
        ref_nvfp4_quant,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_mfma_gemm import (
        NVFP4_BLOCK_SIZE,
        gemm_nvfp4_mfma,
        quantize_nvfp4_activation,
    )

    torch.manual_seed(0)
    dev = "cuda"
    bs = NVFP4_BLOCK_SIZE

    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=dev)

    # Per-tensor global scales (== 1 / divisor stored in checkpoints).
    a_global = (x.abs().max() * FLOAT4_E2M1_MAX_RECIPROCAL).to(torch.float32)
    w_global = (w.abs().max() * FLOAT4_E2M1_MAX_RECIPROCAL).to(torch.float32)
    a_global_inv = (1.0 / a_global).to(torch.float32)
    w_global_inv = (1.0 / w_global).to(torch.float32)
    alpha = (a_global * w_global).to(torch.float32)

    # Quantize weight to NVFP4 (packed E2M1 + E4M3 block scales).
    w_fp4_packed, w_sf = ref_nvfp4_quant(w.to(torch.float32), w_global_inv, bs)
    # ref_nvfp4_quant returns fp32 fp4 values + fp8 scales; repack to uint8.
    from vllm.model_executor.layers.quantization.utils.nvfp4_mfma_gemm import (
        _pack_e2m1,
    )

    w_fp4 = _pack_e2m1(w_fp4_packed.reshape(n, k))
    w_sf = w_sf.view(torch.float8_e4m3fn)

    # MFMA kernel path.
    x_fp4, x_sf = quantize_nvfp4_activation(x, a_global_inv, block_size=bs)
    y_mfma = gemm_nvfp4_mfma(
        x_fp4=x_fp4,
        w_fp4=w_fp4,
        a_sf=x_sf,
        w_sf=w_sf,
        alpha=alpha,
        out_dtype=torch.bfloat16,
    )

    # Emulation reference: dequant both operands to bf16 and matmul.
    x_dq = _quant_dequant(x, a_global_inv, bs)
    w_dq = dequantize_to_dtype(
        w_fp4.view(torch.uint8),
        w_sf,
        w_global_inv,
        torch.bfloat16,
        bs,
        swizzle=False,
    )
    y_ref = torch.matmul(x_dq, w_dq.t())

    torch.testing.assert_close(
        y_mfma.float(), y_ref.float(), atol=2e-1, rtol=2e-1
    )


def _quant_dequant(x, global_scale, bs):
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        ref_nvfp4_quant_dequant,
    )

    return ref_nvfp4_quant_dequant(x, global_scale, bs)
