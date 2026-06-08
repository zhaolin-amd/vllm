# 在 MI355 (gfx950 / CDNA4) 上用 FP4 MFMA 跑 NVFP4 模型

> 分支: `nvfp4-to-mxfp4-rocm`
> 目标硬件: AMD MI355 (gfx950, CDNA4)，原生支持 OCP-MX (MXFP4)，**不**原生支持 NVFP4。
> 状态: 设计 + 实现 (opt-in，默认行为零改动)。**本机无 GPU/aiter/triton，无法编译验证，首跑需在真机调试。**

## 1. 问题

手上有 **NVFP4 量化好的 checkpoint**，想在 MI355 上部署，要求：

- 速度不能比"native NVFP4"慢很多（注：MI355 上根本没有 native NVFP4，参照系实际是 native MXFP4）。
- 精度要≈NVFP4。

vLLM 在 ROCm 上对 NVFP4 目前只有 emulation 一条路（`EmulationNvFp4LinearKernel`）：把权重整张 dequant 成 BF16 再 `torch.matmul`（见 `nvfp4_emulation_utils.py::run_nvfp4_emulations`）。它**完全不碰 FP4 matrix core**，所以很慢。

## 2. 硬件约束（已用 aiter 源码确认）

参考 `ROCm/aiter`（本地 `/proj/rdi/staff/zhaolin/code/github/aiter`）：

- ROCm 的 native MXFP4 是真 **W4A4**（fp4 激活 × fp4 权重），入口 `gemm_afp4wfp4` / `gemm_a4w4`。
- 底层是 CDNA4 的 scaled-MFMA intrinsic：
  `gl.amd.cdna4.mfma_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc)`
  （`aiter/ops/triton/gluon/gemm_afp4wfp4.py:345`），Triton 层是
  `tl.dot_scaled(a, a_scales, "e2m1", b, b_scales, "e2m1", acc)`
  （`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py:180`）。
- **scale 是 MMA 指令的内部入参**，不是事后乘的标量。`SCALE_GROUP_SIZE = 32` 是 `constexpr`，硬绑 **block-32**；scale 是 **E8M0**（纯 2 的幂，量化 kernel 里 `bs_e8m0 = unbiased + 127`、`exp2(...)`）。`*_format` 只接受 `"e2m1"`。

NVFP4 与之的差异正好卡在 scale 上：**block-16** + **E4M3（带尾数）** + per-tensor fp32 **global scale**。这三点 native scaled-MFMA 通路都给不了 —— 所以才只能 emulate。

### 关键洞察

`tl.dot_scaled` 的 E8M0 scale 喂 **127（= 2^0 = 1）** 时，硬件缩放变成**恒等**。于是这条指令退化成一个**未缩放的 fp4×fp4 → fp32 矩阵乘单元**。我们可以借这个单元的算力，把 NVFP4 真正的 scale（E4M3 / block-16 / global）放到 fp32 软件侧自己施加。这就是方案 B1。

## 3. 方案对比

| 路线 | 用 scaled-MFMA? | 速度 | 精度 | 工作量 |
|---|---|---|---|---|
| 现状 emulation | 否（BF16 matmul） | 很慢 | =NVFP4 | 0 |
| A. 离线转码 NVFP4→MXFP4 | 是，原生 | 满血 | ≈MXFP4（略低于 NVFP4） | 低（离线脚本） |
| **B1. 非缩放 fp4 MMA + 软件 scale** | 借用（scale 喂 1） | 低于 native，远高于 emulation | **=NVFP4** | 高（改 Triton kernel） |
| B2. 把 NVFP4 scale 塞进 E8M0/block-32 | 是 | 满血 | ≈MXFP4 | = 方案 A，无意义 |

本文档实现 **B1**。

## 4. B1 数值推导

单个输出元素，沿 K 累加，把三级缩放拆开：

```
真实权重 = e2m1_w · w_sf16(E4M3) · w_global
真实激活 = e2m1_a · a_sf16(E4M3) · a_global

out[m,n] = alpha · Σ_blocks{ (a_sf16_b · w_sf16_b) · [ Σ_within_16  e2m1_a · e2m1_w ] }
           ^^^^^                ^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        epilogue 一次         fp32 累加里逐 block 施加      FP4 MFMA(硬件, scale=1)

alpha = a_global · w_global   (两个 per-tensor 常数，vLLM 加载时已预算成 layer.alpha)
```

为什么 block scale 必须在 kernel 内、按 block-16 施加：量化分组在 **K（reduction）轴**上。一旦 dot 跨过 block 边界求和，各 block 的贡献就混在一起，事后再乘任何单一标量都无法分离。global scale 是 per-tensor 公因子，可以提到 epilogue。

## 5. Kernel 伪码

```python
# x_fp4: (M, K//2) uint8   w_fp4: (N, K//2) uint8
# a_sf16/w_sf16: E4M3, block=16   alpha: per-tensor 标量
NEUTRAL_E8M0 = 127   # 2^0 = 1

acc = 0.0  # fp32
for kb in range(0, K, 16):                     # NVFP4 block = 16
    p = dot_scaled(x_fp4[:, kb:kb+16], NEUTRAL_E8M0, "e2m1",
                   w_fp4[kb:kb+16],   NEUTRAL_E8M0, "e2m1")   # 未缩放 fp4 乘积
    acc += p * outer(a_sf16[:, kb//16], w_sf16[:, kb//16])    # E4M3 外积，fp32 里施加
out = (acc * alpha).to(out_dtype)
```

相对 aiter `_gemm_afp4wfp4_kernel` 的改动：
1. `SCALE_GROUP_SIZE` 32 → **16**；K 主循环步长相应改。
2. 不再从 `a_scales_ptr`/`b_scales_ptr` 读 E8M0 喂 `dot_scaled`，而是喂**常数 127**。
3. 读 E4M3 的 `a_sf16`/`w_sf16`（view 成 fp8_e4m3），在 fp32 累加器里乘**外积**（`[M,1] × [1,N]`）。
4. epilogue 乘 `alpha`。

## 6. 代码改动清单

全部 **opt-in**，默认路径零改动。

1. **新 Triton kernel** —
   `vllm/model_executor/layers/quantization/utils/nvfp4_mfma_gemm.py`
   - `gemm_nvfp4_mfma(x_fp4, w_fp4, a_sf16, w_sf16, alpha, out_dtype)`：上述 kernel + Python wrapper。
   - 激活量化 helper：per-1×16 + E4M3 + a_global（复用 `ref_nvfp4_quant` 逻辑，`nvfp4_emulation_utils.py:439`）。
   - 顶部 `try: import triton ... except`，import 失败时 kernel 不可用（由 kernel 类的 `is_supported` 兜底）。

2. **新 kernel 类** —
   `vllm/model_executor/kernels/linear/nvfp4/nvfp4_mxfp4_mfma.py`
   `Nvfp4Mxfp4MfmaLinearKernel(NvFp4LinearKernel)`，flow 名 `nvfp4-mxfp4-mfma`。
   - `is_supported`: `current_platform.is_rocm()` 且 `on_gfx950()` 且 triton 可用。
   - `process_weights_after_loading`: 权重保持 NVFP4 packed + E4M3 block scale 原样；打印启动横幅明确告知用户走的是此 flow 而非 emulation。
   - `apply_weights`: 激活量化 → 调 `gemm_nvfp4_mfma` → 加 bias。

3. **注册** — `vllm/model_executor/kernels/linear/__init__.py`
   - import + 加入 `_NVFP4_BACKEND_TO_KERNEL["nvfp4-mxfp4-mfma"]`。
   - **不**加入 `_POSSIBLE_NVFP4_KERNELS[PlatformEnum.ROCM]` 的自动选择列表 —— 纯 opt-in，ROCm 默认仍为 emulation，零行为改动。

4. **测试** —
   `tests/kernels/quantization/test_nvfp4_mfma_gemm.py`
   - 与 `run_nvfp4_emulations` 对拍，断言数值接近（atol/rtol 容忍 fp4 量化误差）。
   - `@pytest.mark.skipif` 非 gfx950 跳过。

## 7. 接入点（已确认）

- NVFP4 scheme: `CompressedTensorsW4A4Fp4`（`compressed_tensors_w4a4_nvfp4.py:28`）通过
  `init_nvfp4_linear_kernel()` 选 kernel。加载侧已就绪、可直接复用：
  - weight global scale 取倒数：`compressed_tensors_w4a4_nvfp4.py:106-109`
  - input global scale + 预算 `layer.alpha`：`:121-133`
  - q/k/v 融合层 global 不一致 warn：`:96-103`
- kernel 选择/注册：`vllm/model_executor/kernels/linear/__init__.py`
  - `init_nvfp4_linear_kernel`：`:851`
  - `_NVFP4_BACKEND_TO_KERNEL`、`_POSSIBLE_NVFP4_KERNELS`（ROCm 自动选择列表仅 `EmulationNvFp4LinearKernel`）。
- kernel 基类/模板：`nvfp4/base.py`、`nvfp4/emulation.py`。

## 8. 启用方式与用户可见性

- **默认（ROCm，含 gfx950）**：emulation。本改动**不修改**任何现有默认。
- **启用此 flow**：`VLLM_NVFP4_GEMM_BACKEND=nvfp4-mxfp4-mfma`（仅 gfx950+triton 支持，否则
  `is_supported()` 报错说明原因）。
- **如何确认走了哪条**：加载时日志区分两者 ——
  - 此 flow：`Running ROCm NVFP4 via the 'nvfp4-mxfp4-mfma' flow: ... This is not the emulation backend.`
  - emulation：选择器原有的 `Using EmulationNvFp4LinearKernel for NVFP4 GEMM` /
    “falling back to the slow and unoptimized emulation backend”。

### 对现有路径的影响（已验证为零）

本改动是纯增量：新增 4 个文件 + 在 `__init__.py` 仅新增 import、backend dict 一项。
未触碰任何 MXFP4 代码、未改动任何现有 NVFP4 kernel、CUDA 选择列表不变、ROCm
自动选择列表不变。因此 MXFP4（native / emulation）与 NVFP4（native / emulation）
四条现有路径行为完全不变。

## 9. 预期与诚实边界

- **精度**：权重保留 NVFP4 原始 block-16+E4M3+global（数据/ scale 不动），激活按 NVFP4 量化 → 数值**严格等于 NVFP4**。与 emulation 不会 bit-identical（累加顺序/中间精度不同），但属同一数值水平。
- **性能**：block-16 打断 K 会降低 MFMA 的 K 利用率（前期估计 ~50% 量级，取决于硬件最小 fp4 MMA 的 K 粒度）+ 每 16-K 一次 fp32 外积开销。**远快于现状 emulation**（后者根本不上 FP4 tensor core），但**慢于 native MXFP4**。具体数字需真机 benchmark。
- **未验证项**（首跑需在 MI355 上调）：
  1. `tl.dot_scaled` 喂常数 127 是否被 lower 成真正的恒等缩放、张量形状是否合法。
  2. E4M3 block-16 scale 的 swizzle/布局是否匹配 kernel 读取。
  3. K 必须能被 16 整除（NVFP4 group_size=16 已保证），tile 配置需对齐到 16。
  4. block-16 截断下的实际 MFMA 利用率与端到端吞吐。

## 10. 备选

若 B1 的 MFMA 利用率损失过大，退回**方案 A（离线转码 NVFP4→MXFP4）**：用 `dequantize_to_dtype` 还原 BF16 → `dynamic_mxfp4_quant` 重量化 → 存 compressed-tensors MXFP4，部署走原生满速 MXFP4，精度≈MXFP4。两条路可并存，按 eval gap 取舍。
