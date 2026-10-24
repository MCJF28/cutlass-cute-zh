# CuTe 密集矩阵乘法 (GEMM) 完整实战教程

> [!NOTE] 原文出处
> 本文翻译自 NVIDIA CUTLASS 官方核心实战教程 [`media/docs/cpp/cute/0x_gemm_tutorial.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0x_gemm_tutorial.md)。配套可运行源码位于官方仓库 [`examples/cute/tutorial/`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/)。

本教程通过剖析一系列自包含、单文件的完整工程示例，手把手带你掌握**仅使用 CuTe 原语从零构建达到硬件峰值性能的密集矩阵乘法（GEMM）内核**。

---

## 1. 基础实战：`sgemm_1.cu`

`sgemm_1.cu` 展示了现代 GPU GEMM 算子的三大基石：
1. **全局显存分块（CTA Tiling）**：将全局大矩阵切分为分配给各个 Thread Block（CTA）的数据分块；
2. **线程协同分块（Thread Partitioning）**：将每个 CTA 内部的数据进一步切分分配给该 Block 内的具体线程；
3. **主循环计算流水线（Mainloop）**：利用 `cute::copy` 与 `cute::gemm` 构建高效的搬运与计算循环。

---

### 1.1 内核高阶接口与泛型参数

内核入口函数 `gemm_device` 签名如下：

```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta);
```

#### 参数语义拆解：
* `ProblemShape`：宏观矩阵乘计算规模 $(M, N, K)$。
* `CtaTiler`：CuTe [分块器概念](./02_layout_algebra.md#22-依模式复合-by-mode-composition-与-tiler)，定义单个 CTA 一次处理的数据块大小。
* `TA / TB / TC` 与 `dA / dB / dC`：各矩阵数据指针与对应的步长。
* `ASmemLayout / BSmemLayout`：每个 CTA 内部用于暂存 $A$ 和 $B$ 分块的共享内存布局。
* `AThreadLayout / BThreadLayout / CThreadLayout`：各个执行阶段用于数据切分分配的线程排布布局。
* `Alpha / Beta`：经典 BLAS 常量：$C = \alpha (A \times B) + \beta C$。

---

### 1.2 全局张量建模：重塑传统 BLAS 认知

传统 BLAS 规范中关于“转置（T）/ 非转置（N）”以及 $M \times K$ 与 $K \times N$ 的复杂混淆，常常使初学者陷入索引深渊。

> [!TIP] CuTe 的跨时代统一视角：M-major、N-major 与 K-major
> CuTe 彻底抛弃了行/列优先与转置的模糊概念，统一采用**物理步长连续性（Majorness）**与**统一坐标语义**：
> * 矩阵 $A$ 的语义永远是 **$(M, K)$**；
> * 矩阵 $B$ 的语义永远是 **$(N, K)$**；
> * 矩阵 $C$ 的语义永远是 **$(M, N)$**；
> * **归约求和轴 $K$ 永远处于输入矩阵的第 2 维！** 算法内核永远只对两个输入矩阵的第二个维度执行内积归约，实现完全对称的通用处理。

| 传统 BLAS 记法 | A 连续主轴 | A 物理布局 | B 连续主轴 | B 物理布局 |
| :---: | :---: | :---: | :---: | :---: |
| **NT** | **M-major** (步长为 1) | `(M, K):(1, ldA)` | **N-major** (步长为 1) | `(N, K):(1, ldB)` |
| **TN** | **K-major** (步长为 1) | `(M, K):(ldA, 1)` | **K-major** (步长为 1) | `(N, K):(ldB, 1)` |
| **NN** | **M-major** | `(M, K):(1, ldA)` | **K-major** | `(N, K):(ldB, 1)` |
| **TT** | **K-major** | `(M, K):(ldA, 1)` | **N-major** | `(N, K):(1, ldB)` |

在设备内核中直接构造全局张量：
```cpp
Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M, K)
Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N, K)
Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M, N)
```

---

### 1.3 CTA 线程块级数据平铺 (CTA Partitioning)

首先定义单个 CTA 处理的数据块大小（编译期静态常量）：
```cpp
auto bM = Int<128>{};
auto bN = Int<128>{};
auto bK = Int<  8>{};
auto cta_tiler = make_shape(bM, bN, bK); // (128, 128, 8)
```

每个 CTA 根据自己的内置坐标 `(blockIdx.x, blockIdx.y, _)`，利用 **`local_tile`** 提取属于自己的全局切片：

```cpp
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m, n, k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,  X, _1>{}); // (BLK_M, BLK_K, k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{}); // (BLK_N, BLK_K, k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{}); // (BLK_M, BLK_N)
```

* `Step` 机制充当投影过滤器（`X` 表示丢弃不相干维度）。
* 对于矩阵 $A$，最终得到一个 Rank-3 的张量 `(BLK_M, BLK_K, k)`，前两维是本 Block 单次迭代的数据块，最后一维 $k$ 则供主循环迭代遍历求和！

---

### 1.4 共享内存 (SMEM) 张量申请与布局

在内核中使用静态布局申请共享内存缓冲池：

```cpp
// 申请共享内存物理存储
__shared__ TA smemA[cosize_v<ASmemLayout>];
__shared__ TB smemB[cosize_v<BSmemLayout>];

// 绑定为 CuTe 强类型打标张量
Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K)
Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K)
```

> [!NOTE] 为什么使用 `cosize`？
> 布局是一个映射函数，其值域的最大范围被称为 `cosize`。使用 `cosize_v` 申请数组，能够确保即便布局内部存在 Padding（填充间隙用于消解 Bank 冲突），申请的显存也绝对能容纳所有有效偏移！

---

### 1.5 搬运阶段线程分区：`local_partition`

如何让 CTA 内的 256 个线程协同将全局显存的 `(128, 8)` 分块搬运至共享内存？
定义线程排布：`auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));`（32 行 $\times$ 8 列，总计 256 线程）。

使用 **`local_partition`** 将张量切分给各个线程：

```cpp
Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (THR_M, THR_K, k)
Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M, THR_K)
```

* `tAgA` 与 `tAsA` 的命名规范：**“以线程模式 `tA` 作用在张量 `gA` / `sA` 上的分区视图”**。
* 每个线程获得一个 `(4, 1)` 大小的独立子张量。
* 协同数据加载仅需一行优雅的调用：
  ```cpp
  copy(tAgA(_, _, k_tile), tAsA);
  ```

---

### 1.6 计算阶段线程分区与主循环 (Mainloop)

在计算矩阵乘时，线程往往需要不同的分工（例如 256 线程按 $16 \times 16$ 排布去瓜分 $128 \times 128$ 的输出矩阵 $C$）。
定义计算线程布局：`auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));`。

```cpp
// 对 sA 投影行模式，对 sB 投影列模式，对 gC 投影完整网格
Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1,  X>{}); // (THR_M, BLK_K)
Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X, _1>{}); // (THR_N, BLK_K)
Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{}); // (THR_M, THR_N)

// 在寄存器堆中分配专属累加器
Tensor tCrC = make_tensor_like(tCgC); // (THR_M, THR_N)
clear(tCrC); // 初始化清零
```

#### 完整的主循环执行流 (Mainloop Pipeline)：

```cpp
auto K_TILE_MAX = size<2>(tAgA);

for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
{
  // 1. 全局显存协同搬运至共享内存 (tA/tB 线程分区)
  copy(tAgA(_, _, k_tile), tAsA);
  copy(tBgB(_, _, k_tile), tBsB);

  // 2. 异步指令等待与显存同步
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads(); // 确保全块线程完成共享内存写入

  // 3. 核心矩阵乘累加计算 (tC 线程分区)
  gemm(tCsA, tCsB, tCrC); // 累加至寄存器
  __syncthreads(); // 确保全块线程完成共享内存读取，方可进入下一轮覆写
}

// 4. 尾声段 (Epilogue): 写回全局显存
axpby(alpha, tCrC, beta, tCgC);
```

---

## 2. 进阶升级：`sgemm_2.cu` 与硬件原语加速

在进阶版本中，CuTe 引入了更强大的 **`TiledCopy`** 与 **`TiledMMA`**：

### 2.1 128 位宽硬件向量化拷贝 (`TiledCopy`)
```cpp
// 强制将元素以 128 位宽 (uint128_t) 向量化拷贝
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint128_t>, TA>{},
    Layout<Shape<_32, _8>>{},  // 线程网格 32x8
    Layout<Shape< _4, _1>>{}); // 每个线程负责 4x1 个 TA (正好 128 位！)

ThrCopy thr_copy_a = copyA.get_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA); // 源端分区 (Global)
Tensor tAsA = thr_copy_a.partition_D(sA); // 目的端分区 (Shared)

// 执行单条 128 位宽直通搬运
copy(copyA, tAgA(_, _, k_tile), tAsA);
```

### 2.2 共享内存 Padding 消除 Bank Conflict
为了彻底解决共享内存访问时的 Bank 冲突，仅需修改共享内存布局的 Stride 定义（无需触碰任何内核内部逻辑）：

```cpp
// 在 M 维度附加 1 个元素的 Padding：
auto sA = make_layout(make_shape (      bM,           bK),
                      make_stride(Int<1>{}, bM + Int<1>{}));
```
跨步错位后，所有线程的访存请求被打散至不同的 Bank，冲突彻底归零！

---

## 3. 终极奥义：GETT 即 GEMM (高维张量缩并)

在现代大模型中，注意力机制（Attention）、专家并行（MoE）往往涉及四维甚至五维张量（如 Batch, Heads, Sequence, Hidden）。
**CuTe 允许将任意多维张量折叠为复合模式（Multi-mode）！**

```cpp
// 将多维 (m0, m1) 聚为一个复合 M 维度：
auto M = make_shape(m0, m1);
auto prob_shape = make_shape(M, N, K);

// 对应的 CTA Tiler 同样采用复合分块：
auto bM = Shape<_64, _2>{}; // m0 取 64，m1 取 2
auto cta_tiler = make_shape(bM, bN, bK);
```

**底层执行内核完全不需要重写任何一行代码！**
无需显式的 Reshape 内存拷贝，高维张量缩并（GETT）直接在硬件上以媲美原生 GEMM 的理论峰值 TFLOPS 全速狂飙！
