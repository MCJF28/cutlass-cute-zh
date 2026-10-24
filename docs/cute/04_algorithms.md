# CuTe 泛型张量算法体系 (Algorithms)

> [!NOTE] 原文出处
> 本文翻译自 NVIDIA CUTLASS 官方核心规范文档 [`media/docs/cpp/cute/04_algorithms.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/04_algorithms.md)。

本文系统梳理作用于 `Tensor` 上的通用数值算法接口与实现机制。所有算法源码均可在官方头文件目录 [`include/cute/algorithm/`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/) 中查阅。

---

## 1. 数据搬运算法：`copy` (Tensor Copy)

`copy` 是 GPU Kernel 中执行最频繁的基础操作，负责将源张量 `src` 的数据高效复制到目的张量 `dst` 中。

头文件：[`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp)。

### 1.1 接口定义与编译期特化调度

由于 CuTe 的 `Tensor` 在编译期封装了**数据类型（Value Type）、存储介质空间（Memory Space Tag）、静态形状与步长**，这为编译器提供了极为强大的编译期推导依据。`copy` 会根据参数的具体类型特征，自动分发（Dispatch）到最适合当前 GPU 微架构的同步或异步硬件专用拷贝指令！

`copy` 拥有两种核心调用签名：

```cpp
// 1. 双参数默认重载: 根据 src 和 dst 的类型特征自动推导最优策略
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE void
copy(Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);

// 2. 带 Copy_Atom 的显式重载: 由调用者显式指定非默认的硬件拷贝原子
template <class... CopyArgs,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE void
copy(Copy_Atom<CopyArgs...>       const& copy_atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>      & dst);
```

---

### 1.2 并发粒度与同步语义 (Concurrency & Synchronization)

无论是默认实现还是由 `Copy_Atom` 指定的实现，其并发行为与同步语义严格取决于参数类型：

* **线程并发粒度**：
  * 单线程串行执行（Per-thread Sequential）；
  * 多线程/线程块/线程集群（Cluster）协同执行（Parallel across threads）。
* **硬件级同步要求**：
  * 若 `copy` 是多线程协同操作，在后续代码读取拷贝结果前，开发者必须插入显式同步原语（如 `__syncthreads()` 或 Cooperative Groups 栅障）。
  * 若 `copy` 底层启用了**异步硬件拷贝指令（如 Ampere 的 `cp.async` 或 Hopper 的 TMA）**，开发者必须按照硬件流水线规范执行专门的异步等待（如 `cp.async.wait_group` 或 TMA 屏障事务），否则将发生读后写（RAW）数据冒险！

---

### 1.3 极简通用实现 vs 硬件级极致优化

在最抽象的理论层面上，一个朴素的通用 `copy` 仅需按一维逻辑坐标遍历：

```cpp
template <class TA, class ALayout, class TB, class BLayout>
CUTE_HOST_DEVICE void
copy(Tensor<TA, ALayout> const& src, Tensor<TB, BLayout>& dst)
{
  for (int i = 0; i < size(dst); ++i) {
    dst(i) = src(i);
  }
}
```

但在现代 GPU 上，CuTe 的优化版 `copy` 会在编译期执行一系列惊艳的微架构优化：
1. **自动向量化合并（Element Vectorization）**：
   * 若输入张量拥有静态布局，且内存对齐满足要求，CuTe 会自动将连续的 4 个 32 位读写操作（`ld.global.b32`）合并为单条 128 位宽指令（`ld.global.b128`），直接打满 GPU 显存总线带宽！
2. **异步硬件指令直通**：
   * 若检测到源指针带有 `gmem_ptr`（全局内存）、目的指针带有 `smem_ptr`（共享内存），自动分发到专用的 `cp.async` 指令，绕过寄存器直接将数据压入 Shared Memory。

---

## 2. 边界保护拷贝：`copy_if` (Predicated Copy)

头文件：[`include/cute/algorithm/copy.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/copy.hpp)。

在工业级矩阵乘法中，当矩阵的实际维度无法被分块大小整除时，边界线程会读写到越界区域。
`copy_if` 引入了一个与输入输出形状完全同构的**谓词张量（Predication Tensor）**：

$$\text{copy\_if}(\text{pred}, \text{src}, \text{dst})$$

仅当谓词张量中对应位置的布尔值为非零（`true`）时，才会执行实际的内存读取与写入，从硬件指令级规避越界异常（非法显存访问 Crash）。详见 [谓词判定专项指南](./0y_predication.md)。

---

## 3. 核心计算算子：`gemm` (Matrix Multiply-Accumulate)

### 3.1 模式表示法与计算范式

`gemm` 算法接收三个张量操作数：输入张量 $A$、$B$，以及累加输出张量 $C$。
CuTe 使用字母标记各张量维度的代数语义：

* **$V$**：向量维度（Vector mode），表示彼此独立的元素；
* **$M, N$**：输出矩阵 $C$ 的行数（Rows）与列数（Columns）；
* **$K$**：GEMM 的**归约求和维度（Reduction mode）**，即沿着该轴执行内积求和。

CuTe 采用记法 `(A 的模式) x (B 的模式) => (C 的模式)` 表达 5 大计算范式：

| 计算范式 | 形式定义 | 硬件底层映射说明 |
| :--- | :--- | :--- |
| **1. 向量逐元素积** | `(V) x (V) => (V)`<br>$C_v \mathrel{+}= A_v \cdot B_v$ | 直接映射到底层 FMA 或单条 MMA 指令 |
| **2. 向量外积** | `(M) x (N) => (M, N)`<br>$C_{mn} \mathrel{+}= A_m \cdot B_n$ | 相当于范式 (4) 中 $V=1$ 的特化形式 |
| **3. 标准矩阵乘** | `(M, K) x (N, K) => (M, N)`<br>$C_{mn} \mathrel{+}= \sum_k A_{mk} \cdot B_{nk}$ | 针对每个 $K$ 维度循环调用范式 (2) 外积累加 |
| **4. 批量向量外积** | `(V, M) x (V, N) => (V, M, N)`<br>$C_{vmn} \mathrel{+}= A_{vm} \cdot B_{vn}$ | **最大化寄存器复用！** 针对每个 $M, N$ 批量分发到范式 (1) |
| **5. 批量分块矩阵乘** | `(V, M, K) x (V, N, K) => (V, M, N)`<br>$C_{vmn} \mathrel{+}= \sum_k A_{vmk} \cdot B_{vnk}$ | **Tensor Core 的核心执行形态！** 针对每个 $K$ 批量迭代 |

> [!TIP] 维度排序约定
> 在 CuTe 规范中：
> * 归约维 $K$ 如果出现，永远处于**最右侧（最外层）**；
> * 向量维 $V$ 如果出现，永远处于**最左侧（最内层）**。

### 3.2 硬件特化调度与 `MMA_Atom`
与 `copy` 类似，`gemm` 同样支持传入可选的 `MMA_Atom` 参数，允许开发者显式覆盖默认的 FMA 指令，强制调用特定 GPU 硬件架构的 Tensor Core 原语（如 `mma.sync.aligned.m16n8k16.row.col`）。详见 [MMA Atom 硬件抽象专项指南](./0t_mma_atom.md)。

---

## 4. 辅助线性代数算法 (Auxiliary Algorithms)

* **`axpby`**（[`include/cute/algorithm/axpby.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/axpby.hpp)）：
  * 计算广义线性组合：$y \leftarrow \alpha x + \beta y$，广泛用于 GEMM 尾声段（Epilogue）对累加结果进行缩放与残差融合。
* **`fill`**（[`include/cute/algorithm/fill.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/fill.hpp)）：
  * 使用指定的标量值覆盖填充目标张量的所有元素。
* **`clear`**（[`include/cute/algorithm/clear.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm/clear.hpp)）：
  * 将目标张量的数据快速清零（在初始化 GEMM 累加器寄存器时极高频使用）。
