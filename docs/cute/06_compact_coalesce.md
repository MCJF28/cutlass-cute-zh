# 06. 布局压缩与对偶运算 (Compact, Coalesce & Upcast)

> **重要免责声明**：本网站只是用来消耗反重力额度的，并且给有需要的人看一看，准确度极低。可用度极低，不想看的请不看，造成错误概不负责。
> 本篇由 AI 辅助汉化与深度解析，非 NVIDIA 官方技术文档。涉及 GPU 底层硬件架构参数与生产算子调优，请以 [NVIDIA CUTLASS 官方源码](https://github.com/NVIDIA/cutlass) 为准。

---

## 1. 为什么需要布局压缩与对偶？

在高性能 GPU 算子开发中，张量的逻辑维度（Shape）往往为了满足算法描述需求而被切分为多维层次结构（例如 `(TileM, TileN, ClusterM, ClusterN, Stage)`）。然而，底层物理硬件（全局内存 GMEM、共享内存 SMEM、寄存器堆 RF 以及硬件 DMA/TMA 引擎）只认识**一维物理地址空间**与**向量化访存连续段**。

如果一个高维张量在某些维度之间是物理连续的，但编译器无法在类型系统中将其识别为一维连续区间，就必须产生大量的坐标换算指令与细粒度的标量访存操作，严重拖垮指令吞吐。

为此，CuTe 提供了一整套基于严格代数公理的**布局压缩（Compact）、合并化简（Coalesce）与位宽对偶变换（Upcast/Downcast）**算子，使得开发者能够在高维优雅抽象与底层极致连续硬件指令之间自由穿梭。

---

## 2. 布局的紧凑性（Compactness）代数公理

在深入算子之前，必须先明确 CuTe 中“紧凑（Compact）”的形式化定义。

### 2.1 紧凑性（Compactness）

设布局 $L: \mathcal{C} \to \mathcal{I}$ 为坐标域到索引域的映射函数，其中坐标总元素个数为 $\text{size}(L)$，映射后的物理地址跨度为：
$$\text{span}(L) = \max_{c \in \mathcal{C}} L(c) - \min_{c \in \mathcal{C}} L(c) + 1$$

根据单射与满射性质，CuTe 将布局分为三个级别：
1. **严格紧凑（Strictly Compact / Bijective）**：
   $$\text{size}(L) = \text{span}(L) \quad \text{且} \quad \forall c_1 \neq c_2, L(c_1) \neq L(c_2)$$
   这意味着输入域的每一个逻辑坐标都一一对应物理内存中的连续无空隙槽位，没有任何重叠，也没有任何浪费的洞穴（Holes）。
2. **多对一紧凑（Surjective Compact / Non-injective）**：
   例如存在步长为 0 的广播维度（Broadcast）。张量在逻辑上具有多个坐标，但在物理上复用相同单元。
3. **稀疏布局（Non-compact / Has Holes）**：
   $\text{span}(L) > \text{size}(L)$，由于存在 Padding 或大跨步切片，物理内存中散落着未被逻辑坐标覆盖的无效空洞。

CuTe 提供了内置特征判断工具：
```cpp
auto layout = make_layout(Shape<_4, _8>{}, Stride<_1, _8>{}); // 稀疏，Stride 跨度为 8 而不是 4
static_assert(is_compact<decltype(layout)>::value == false, "Layout has holes!");
```

---

## 3. 布局合并化简：`coalesce`

`coalesce`（合并）是 CuTe 优化器中最频繁调用的算子之一。它的核心目标是：**在不改变任何逻辑坐标到物理地址映射关系的前提下，尽可能合并物理连续的相邻维度，消除冗余的层次结构。**

### 3.1 维度折叠代数法则

考虑两个相邻的逻辑模式 $(d_i, d_{i+1})$，其对应的形状为 $(S_i, S_{i+1})$，步长为 $(D_i, D_{i+1})$。
若满足连续性充要条件：
$$D_{i+1} = S_i \times D_i$$
则这两个维度在物理内存中是完全首尾相接、密铺连续的！因此可以将它们直接折叠合并为一个单维度：
$$S'_{i} = S_i \times S_{i+1}, \quad D'_{i} = D_i$$

### 3.2 典型示例演示

```cpp
#include <cute/tensor.hpp>
using namespace cute;

// 原始布局：嵌套的三维结构
auto orig_layout = make_layout(make_shape(_2{}, _4{}, _8{}), 
                               make_stride(_1{}, _2{}, _8{}));
// 验证连续性：
// dim 0: size 2, stride 1 -> 跨度 2*1 = 2
// dim 1: size 4, stride 2 -> 跨度 4*2 = 8
// dim 2: size 8, stride 8 -> 跨度 8*8 = 64

// 调用 coalesce 进行维度合并
auto compact_layout = coalesce(orig_layout);

// 打印结果：直接合并为一维扁平布局
print(compact_layout); 
// 输出: (_64):(_1)
```

在内核生成中，当 `coalesce` 将多维布局折叠为 `(_64):(_1)` 后，后续的数据搬运算法（如 `cute::copy`）会自动识别到这是一个 64 元素的连续内存段，从而自动提升为一条 128 位的向量化加载指令，而不是发射 64 条孤立的标量读取指令！

---

## 4. 自动构建紧凑布局：`compact_col_major` 与 `compact_row_major`

当算法确定了某种多维分块形状（Shape），但需要我们为该形状赋予一套最致密、零空洞的连续物理步长时，CuTe 提供了 `compact_col_major` 与 `compact_row_major` 生成器。

### 4.1 列优先紧凑构建 (`compact_col_major`)

按反向字典序（从左向右）前向累乘步长：
$$D_0 = 1, \quad D_{k} = D_{k-1} \times S_{k-1}$$

```cpp
auto shape = make_shape(_4{}, make_shape(_2{}, _3{}));
auto l_col = compact_col_major(shape);
// 计算过程：
// mode 0: size 4 -> stride 1
// mode 1.0: size 2 -> stride 4
// mode 1.1: size 3 -> stride 4 * 2 = 8
// 得到 Layout: (_4, (_2, _3)) : (_1, (_4, _8))
```

### 4.2 行优先紧凑构建 (`compact_row_major`)

按正向字典序（从右向左）反向累乘步长：

```cpp
auto shape = make_shape(_4{}, _8{});
auto l_row = compact_row_major(shape);
// 得到 Layout: (_4, _8) : (_8, _1)
```

---

## 5. 位宽对偶变换：`upcast` 与 `downcast`

在现代混合精度计算（例如 FP8 GEMM、INT4 量化网络或 FP32 累加）中，同一个内存缓冲区经常需要在不同的数据类型视界之间切换。例如：
- 在全局内存搬运阶段，以 `uint128_t`（16 字节）为单元进行向量化传输；
- 在 Tensor Core 计算阶段，以 `half`（2 字节）或 `fp8`（1 字节）为单元参与矩阵乘；
- 在量化反量化阶段，以 `uint32_t` 提取包含 8 个 INT4 权重的压缩包。

`upcast<N>` 与 `downcast<N>` 允许我们**直接对布局进行位宽伸缩运算**，无需重新编写繁琐的多维索引换算。

### 5.1 数据类型升级：`upcast<N>(Layout)`

设缩放比率 $N$ 表示目标数据类型的位宽是原始数据类型的 $N$ 倍（例如从 `half` 到 `uint128_t`，每个单元包含 8 个 half，则 $N = 8$）。
`upcast<N>` 的数学行为如下：
1. 它要求布局的最内层连续维度的形状必须能够被 $N$ 整除，且步长必须为 1；
2. 将最内层维度的形状除以 $N$；
3. 将所有外部维度的步长除以 $N$。

```cpp
// 原始布局：存储 half 类型的 8 行 16 列矩阵，列优先
auto layout_half = make_layout(make_shape(_8{}, _16{}), make_stride(_1{}, _8{}));

// 升级视界：以 128-bit 向量单元 (包含 8 个 half, N=8) 重新观察该内存
auto layout_vec = upcast<8>(layout_half);

// 结果验证：
// 原第 0 维 size 8 变为 8/8 = 1
// 原第 1 维 stride 8 变为 8/8 = 1
// 得到布局：(_1, _16) : (_1, _1) -> coalesce 后折叠为 (_16):(_1)
print(coalesce(layout_vec));
```

### 5.2 数据类型降级：`downcast<N>(Layout)`

与 `upcast` 相反，`downcast<N>` 用于将高位宽单元拆解为细粒度的低位宽单元（例如将 4 字节的 `uint32_t` 解构成 8 个 4-bit 的 INT4 元素）。
- 它在最内层维度插入一个形状为 $N$、步长为 1 的全新子维度；
- 将原有维度的步长同乘以 $N$。

```cpp
// 原始寄存器布局：4 个 uint32_t
auto layout_u32 = make_layout(Shape<_4>{}, Stride<_1>{});

// 降级视界：每个 uint32_t 包含 2 个 half (N=2)
auto layout_fp16 = downcast<2>(layout_u32);

// 结果：Shape 变为 (_2, _4)，Stride 变为 (_1, _2)
print(layout_fp16);
```

---

## 6. 总结与实践范式

| 操作算子 | 核心代数功能 | 典型工业级应用场景 |
| :--- | :--- | :--- |
| `coalesce` | 消除相容维度的阶数，极大化连续片段 | 数据搬运指令自动向量化匹配 |
| `compact_col_major` | 构造保序的最致密列优先物理步长 | 共享内存缓冲区与中间张量动态分配 |
| `upcast<N>` | 视界位宽放大 $N$ 倍，折叠内层维度 | 将标量张量提升为 128 位 `cp.async` 拷贝单元 |
| `downcast<N>` | 视界位宽缩小 $N$ 倍，展开内层细粒度 | 大模型量化（W4A16 / W8A8）权重实时解包计算 |
