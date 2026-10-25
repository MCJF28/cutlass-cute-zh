# 27. 混合专家模型 (MoE) 分组 GEMM 架构 (MoE Grouped GEMM)

> **重要免责声明**：本网站只是用来消耗反重力额度的，并且给有需要的人看一看，准确度极低。可用度极低，不想看的请不看，造成错误概不负责。
> 本篇由 AI 辅助汉化与深度解析，非 NVIDIA 官方技术文档。涉及 GPU 底层硬件架构参数与生产算子调优，请以 [NVIDIA CUTLASS 官方源码](https://github.com/NVIDIA/cutlass) 为准。

---

## 1. MoE 混合专家大模型的算子挑战

随着 **Mixtral 8x7B、DeepSeek-V2/V3、DBRX** 等前沿模型的问世，**混合专家架构（Mixture of Experts, MoE）** 已经成为大语言模型突破参数规模与计算预算瓶颈的绝对主流。

在标准密集模型中，所有 Token 都会经过完全相同的多层感知机（MLP/FFN）计算；而在 MoE 架构中：
- 每一层包含 $E$ 个独立的专家子网络（例如 $E = 8, 64, 256$ 个专家）；
- 一个门控路由网络（Gating Router）为每个输入的 Token 动态选出得分最高的 Top-$K$ 个专家（例如 8 选 2，或 256 选 8）；
- 选中的 Token 被分发给对应的专家进行两层矩阵乘计算（Gate+Up 投影与 Down 投影），最后按路由权重加权求和。

```
              动态输入 Tokens (例如 Batch=128)
                           |
                           v
              [ 门控路由网络: Router ]
             /            |           \
            v             v            v
     [ Expert 0 ]   [ Expert 1 ] ... [ Expert 7 ]
       分配 45 Token   分配 12 Token    分配 80 Token
            \             |            /
             v            v           v
           [ 输出加权汇聚: Weighted Sum ]
```

### 1.1 严重负载倾斜（Load Imbalance）与传统计算死穴

由于自然语言语义的非均匀性，各个专家分到的 Token 数量在运行期时刻剧烈变动：
- **方案 A（Padding 静态补齐）**：
  将所有专家的输入强制补齐为最大批次（例如都补齐到 128）。这会导致大量的空白填充，**高达 60% ~ 80% 的算力被白白浪费在与 0 乘加的无效运算中**，显存开销暴增！
- **方案 B（串行循环调用 GEMM）**：
  为每个专家按其实际 Token 数 Launch 一个独立的标准 GEMM（例如调用 $E$ 次 `cublasGemm`）。这会导致极度严重的**内核启动开销（Launch Overhead）**，且每个小 GEMM 的规模太小，根本无法喂饱拥有 132 个 SM 的现代大显卡，GPU 利用率通常惨跌至 20% 以下！

---

## 2. 救赎之道：CUTLASS Grouped GEMM

为了彻底解决 MoE 算子的性能死穴，**CUTLASS 独家研发了划时代的分组矩阵乘架构：Grouped GEMM（或称 Batched Ragged GEMM）**！

### 2.1 什么是 Grouped GEMM？

Grouped GEMM 允许在**单次内核启动（Single Kernel Launch）**中，并发处理一组形状互不相同的独立矩阵乘任务：

$$C_i = A_i \times B_i + C_i, \quad i \in [0, E-1]$$

其中：
- 专家的权重尺寸 $N$ 和 $K$ 通常是静态固定的（例如 $N=4096, K=4096$）；
- 但每个专家的有效 Token 数 $M_i$ 动态可变（$M_0 = 45, M_1 = 12, \dots$），且满足 $\sum_{i=0}^{E-1} M_i = \text{Total\_Tokens} \times \text{TopK}$。

```
                Grouped GEMM 连续压平布局
[ Expert 0: M=45 ] [ Expert 1: M=12 ] [ Expert 2: M=80 ] ...
+----------------+------------------+------------------+ ...
| CTA 并发网格在软件调度器指挥下，自动无缝跨越边界流水线计算   |
+------------------------------------------------------+ ...
```

---

## 3. Grouped GEMM 内部调度微架构

要在单内核内高效调度尺寸不一的多个 GEMM，CUTLASS 设计了精密的内部调度微架构。

### 3.1 瓦片几何总数与前缀和映射

1. **计算每个专家的 Tile 数量**：
   $$\text{Tiles}_i = \left\lceil \frac{M_i}{\text{TileM}} \right\rceil \times \left\lceil \frac{N}{\text{TileN}} \right\rceil$$
2. **计算全局 Tile 前缀和（Prefix Sum）**：
   $$\text{Prefix}[k] = \sum_{j=0}^{k-1} \text{Tiles}_j, \quad \text{Total\_Tiles} = \text{Prefix}[E]$$
3. **CTA 动态二分查找定位**：
   当某个常驻 CTA 领取到全局瓦片编号 $\text{Tile\_ID} \in [0, \text{Total\_Tiles}-1]$ 时，调度器通过超高效的寄存器内二分查找（Binary Search），以几个时钟周期的延迟迅速判定：
   - 当前 Tile 属于哪一个专家 $e$；
   - 在该专家内部的局部 $(M_{\text{local}}, N_{\text{local}})$ 坐标是多少；
   - 提取该专家的输入、权重与输出基地址指针。

### 3.2 两种内存组织范式

1. **指针数组模式（Pointers Array Mode）**：
   输入为一组指针数组 `ElementA* ptr_A_array[]`，适用于各个专家的激活数据散落在显存不同位置的场景；
2. **连续重排模式（Contiguous Ragged Mode）**：
   所有经过 Router 排序后的 Token 紧密拼接在同一片连续的显存张量中，配合偏移量数组（Offset Array）寻址，享有最优的 TMA / 向量化连续访存带宽。

---

## 4. CUTLASS 3.x 实战代码范式

```cpp
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>

using ElementA          = cutlass::half_t;
using ElementB          = cutlass::half_t;
using ElementC          = cutlass::half_t;
using ElementAccum      = float;

// 1. 声明 Grouped GEMM 设备端算子
using GemmGrouped = cutlass::gemm::device::GemmGrouped<
    cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
        ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
        ElementC, cutlass::layout::RowMajor,
        ElementAccum,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm90, // 支持 Hopper / Blackwell
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<ElementC, 8, ElementAccum, ElementAccum>,
        cutlass::gemm::kernel::GroupedGemmTileScheduler // 专用组调度器
    >::GemmKernel
>;

// 2. 运行期准备问题规模列表
int problem_count = num_experts; // 例如 8 或 64
std::vector<cutlass::gemm::GemmCoord> problem_sizes;
for (int i = 0; i < problem_count; ++i) {
  problem_sizes.push_back({tokens_per_expert[i], N, K});
}

// 3. 构建参数并发射单内核
typename GemmGrouped::Arguments args(
    problem_sizes.data(), problem_count,
    threadblock_count,
    epilogue_params,
    ptr_A_array, ptr_B_array, ptr_C_array, ptr_D_array,
    lda_array, ldb_array, ldc_array, ldd_array
);

GemmGrouped gemm;
gemm.initialize(args);
gemm.run(); // 单次 Launch 算完全部 Expert!
```

---

## 5. 性能收益

在主流开源大模型推理引擎（如 vLLM、TensorRT-LLM、SGLang）中采用 CUTLASS Grouped GEMM 后：
- **消灭 95% 以上的内核启动开销**；
- 算子计算吞吐相比 Padding 方案提升 **$2.5\times \sim 4\times$**；
- 成为支撑百万级并发 MoE 模型极致吞吐的绝对性能底座。
