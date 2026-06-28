# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 linear kernel for AMD CDNA4 (gfx950) using the MXFP4 FP4 MFMA engine.

Runs an NVFP4-quantized checkpoint on hardware that only natively supports
OCP MXFP4 (MI355 / gfx950). The FP4 multiply-accumulate is borrowed from the
MXFP4 matrix-core path; NVFP4's E4M3 / block-16 / global scales are applied
in fp32 software. See ``docs/design/nvfp4_on_rocm_mfma.md`` and
``vllm.model_executor.layers.quantization.utils.nvfp4_mfma_gemm``.

This backend is opt-in. It is never auto-selected; enable it explicitly with
``--linear-backend nvfp4_mxfp4_mfma``. Without it, ROCm NVFP4 keeps using the
emulation backend.
"""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

logger = init_logger(__name__)

# User-facing name for this flow, used in the startup banner so it is obvious
# whether ROCm NVFP4 is running emulation or this FP4-MFMA path.
BACKEND_NAME = "nvfp4_mxfp4_mfma"


class Nvfp4Mxfp4MfmaLinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the MXFP4 FP4 MFMA engine, scales applied in software.

    W4A4 on gfx950: borrows the MXFP4 matrix core for the fp4xfp4 product and
    applies NVFP4's own E4M3/block-16/global scales in fp32. Opt-in only.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, f"{BACKEND_NAME} requires ROCm"
        try:
            from vllm.platforms.rocm import on_gfx950
        except ImportError:
            return False, "on_gfx950 unavailable"
        if not on_gfx950():
            return False, f"{BACKEND_NAME} requires gfx950 (CDNA4) FP4 MFMA"
        from vllm.model_executor.layers.quantization.utils.nvfp4_mfma_gemm import (
            nvfp4_mfma_gemm_available,
        )

        if not nvfp4_mfma_gemm_available():
            return False, f"triton is required for the {BACKEND_NAME} kernel"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        # NVFP4 layers are uniform (group_size 16, E4M3 block scales); no
        # per-layer constraint beyond what is_supported already checks.
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logger.info_once(
            "Running ROCm NVFP4 via the '%s' flow: NVFP4 weights on the MXFP4 "
            "FP4-MFMA engine (gfx950) with scales applied in software. This is "
            "not the emulation backend.",
            BACKEND_NAME,
        )
        # Weight stays as NVFP4: packed E2M1 (uint8) + E4M3 block scales.
        # The CompressedTensors scheme already renamed weight_packed ->
        # weight, took the reciprocal of the global scales, and pre-computed
        # layer.alpha = input_global_scale * weight_global_scale.
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.nvfp4_mfma_gemm import (
            NVFP4_BLOCK_SIZE,
            gemm_nvfp4_mfma,
            quantize_nvfp4_activation,
        )

        out_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1])

        # Quantize activation to NVFP4 (E2M1 + E4M3 block scales). The scheme
        # stores the divisor as input_global_scale_inv (== 1 / input_global).
        x_fp4, x_sf = quantize_nvfp4_activation(
            x_2d, layer.input_global_scale_inv, block_size=NVFP4_BLOCK_SIZE
        )

        y = gemm_nvfp4_mfma(
            x_fp4=x_fp4,
            w_fp4=layer.weight,
            a_sf=x_sf,
            w_sf=layer.weight_scale,
            alpha=layer.alpha,
            out_dtype=out_dtype,
        )

        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])
