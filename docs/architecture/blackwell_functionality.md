# NVIDIA Blackwell (SM100) 架构核心特性与前沿算子支持

> [!NOTE] 原文出处
> 本文翻译自 NVIDIA CUTLASS 官方前沿规范文档 [`media/docs/cpp/blackwell_functionality.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)。

NVIDIA Blackwell (SM100) 架构引入了革命性的 **第 5 代 Tensor Core 指令集 —— `tcgen05.mma`**。
新一代指令不仅全面兼容所有传统精度（TF32、FP16、BF16、INT8），更原生引入了 **4-bit、6-bit 与 8-bit 微缩放数据格式（Block Scaled Floating Point / Microscaling）**，其实测计算吞吐量达到 Hopper 架构 WGMMA 的 **2 倍至 4 倍**！

---

## 1. Blackwell 第 5 代 Tensor Core 指令全景

Blackwell SM100 架构原生支持 7 类核心的 `tcgen05.mma` 硬件指令：

| PTX 硬件指令前缀 | 相比 Hopper 理论吞吐提升 | 支持的数据类型与物理排布 |
| :--- | :---: | :--- |
| `tcgen05.mma.cta_group::[1\|2].kind::tf32` | **2x Hopper TF32** | $A(\text{TF32}) \times B(\text{TF32})$，支持 TN, NT, TT, NN 布局 |
| `tcgen05.mma.cta_group::[1\|2].kind::f16` | **2x Hopper FP16** | $A(\text{FP16/BF16}) \times B(\text{FP16/BF16})$ |
| `tcgen05.mma.cta_group::[1\|2].kind::i8` | **2x Hopper INT8** | $A(\text{INT8/UINT8}) \times B(\text{INT8/UINT8})$ |
| `tcgen05.mma.cta_group::[1\|2].kind::f8f6f4` | **2x Hopper FP8** | 混合超低精度：$A(\text{F4/F6/F8}) \times B(\text{F4/F6/F8})$ |
| `tcgen05.mma.kind::mxf8f6f4.block_scale` | **2x Hopper FP8** | 块级微缩放混合精度（OCP MX 规范） |
| `tcgen05.mma.kind::mxf4.block_scale` | **4x Hopper FP8** | 块级微缩放 MXFP4（极致吞吐，专供 TN 布局） |
| `tcgen05.mma.kind::mxf4nvf4.block_scale` | **4x Hopper FP8** | 块级微缩放 MXFP4 与 NVIDIA 原生 NVFP4 |

> [!TIP] 双 SM 硬件协同 (CTA Group 2)
> 观察指令中的 `cta_group::[1|2]` 修饰符：Blackwell 支持由 **2 个物理 SM 组成联合协同组（2SM Cooperative）**，联合驱动单次超大规模硬件矩阵乘，直接突破单 SM 的寄存器与共享内存容量物理天花板！

---

## 2. 块级微缩放矩阵乘 (Block Scaled GEMMs)

在大模型推理中，FP4/FP8 由于位宽过窄极易溢出或下溢。Blackwell 硬件原生支持 **微缩放因子（Scale Factors, SF）** 体系：

$$D = C + (A \times \text{SFA}) \times (B \times \text{SFB})$$

### 2.1 缩放机制与维度映射
* 缩放因子沿着 GEMM 的 **$K$ 归约维度** 施加；
* 密集矩阵中，矩阵 $A$ 和 $B$ 每 **16 个或 32 个元素** 共享 1 个专属缩放因子（在 2:4 结构化稀疏下，由于 $K$ 轴压缩 2 倍，对应 32 或 64 个元素）；
* 若矩阵 $A$ 大小为 $M \times K$，则其对应的缩放因子矩阵 $\text{SFA}$ 大小仅为 $M \times \lceil K/32 \rceil$；
* 硬件单元在执行乘加累加的同时，在数据通路内**以零指令周期开销自动完成定标解算**！

---

## 3. Blackwell 窄精度（Narrow-Precision）数据类型体系

Blackwell 支持极为丰富的超窄精度浮点与缩放类型：

| 数据类型标识符 | 阶码位数 (Exponent) | 尾数位数 (Mantissa) | 是否带符号 (Signed) | 物理总位宽 | 规范说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`float_e4m3_t`** | 4 | 3 | 是 | 8-bit | 标准 FP8 (E4M3) |
| **`float_e5m2_t`** | 5 | 2 | 是 | 8-bit | 标准 FP8 (E5M2) |
| **`float_e2m3_t`** | 2 | 3 | 是 | 6-bit | 窄精度 FP6 |
| **`float_e3m2_t`** | 3 | 2 | 是 | 6-bit | 窄精度 FP6 |
| **`float_e2m1_t`** | 2 | 1 | 是 | 4-bit | 窄精度 FP4 |
| **`float_ue8m0_t`**| 8 | 0 | 否 (纯无符号阶码) | 8-bit | **专用于微缩放因子 (OCP MX 标准)** |
| **`float_ue4m3_t`**| 4 | 3 | 否 | 8-bit | **专用于微缩放因子 (NVIDIA NV 标准)** |

### 3.1 OCP MX 与 NVIDIA NV 规范对照

* **`mx_float4_t` / `mx_float6_t` / `mx_float8_t`**：严格遵循开放计算项目（Open Compute Project, OCP）微缩放规范，缩放向量大小固定为 32，缩放因子精度采用 `float_ue8m0_t`。
* **`nv_float4_t`**：NVIDIA 专有高精度缩放 FP4 格式，缩放向量大小缩短至 16（更细粒度的定标），缩放因子精度采用 `float_ue4m3_t`。

---

## 4. 调度策略与分块尺寸推荐 (Tile Shapes)

在 Blackwell SM100 架构中，CUTLASS 提供了专属的调度策略标签：

* **单 SM 调度**：`KernelTmaWarpSpecialized1SmSm100`（分块推荐：$128 \times 128 \times 128$ 或 $128 \times 256 \times 128$）；
* **双 SM 协同调度**：`KernelTmaWarpSpecialized2SmSm100`（分块推荐：$256 \times 128 \times 128$ 或 $256 \times 256 \times 128$）；
* **2:4 结构化稀疏**：`KernelSparseTmaWarpSpecialized1SmSm100`。

通过 CUTLASS 3.x 统一的 `CollectiveBuilder` 接口，开发者仅需切换精度标签（如将 `half_t` 替换为 `mx_float4_t`），底层即会自动分发并激活 Blackwell 的峰值微缩放硬件引擎！
