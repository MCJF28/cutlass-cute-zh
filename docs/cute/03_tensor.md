# CuTe Tensor 核心抽象与切片分区 (Tensors)

> [!NOTE] 原文出处
> 本文翻译自 NVIDIA CUTLASS 官方核心规范文档 [`media/docs/cpp/cute/03_tensor.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/03_tensor.md)。

本文深入解析 CuTe 的核心容器 —— **`Tensor`（张量）**。
如果说 `Layout` 是一套纯粹的数学映射理论，那么 `Tensor` 则是将这一代数体系真正落地于物理硬件的物理载体。

`Tensor` 抽象表达了一个多维数组，它彻底屏蔽了底层数组元素**如何组织（排布）**以及**存放在何处（内存空间）**的所有繁杂细节。这使开发者能够编写通用的多维张量算法，并针对特定张量属性进行编译期特化（例如针对 Tensor Rank 分发、针对特定内存布局优化、针对数据精度验证等）。

从模板实现来看，`Tensor` 由两个核心参数定义：**`Engine`（存储引擎）** 与 **`Layout`（布局）**：

$$\text{Tensor} = \text{Engine} + \text{Layout}$$

* **`Layout`**：提供张量的多维逻辑形状、访问运算符，并将逻辑坐标映射为物理线性偏移量。
* **`Engine`**：持有一个底层随机访问迭代器（Random-access Iterator）或裸指针包装，根据 `Layout` 计算出的偏移量进行解引用以存取真实数据。
* 底层数据可以驻留在现代 GPU 的任意存储介质中 —— **全局显存（Global Memory）、共享内存（Shared Memory）、本地寄存器堆（Registers）**，甚至可以是在读取时动态计算生成的虚拟数据！

---

## 1. 基础容器操作接口 (Fundamental Operations)

CuTe `Tensor` 提供了类似于标准 STL 容器的操作接口：

* `.data()`：获取该 Tensor 持有的底层原始迭代器或指针。
* `.size()`：获取该 Tensor 的逻辑元素总容量。
* `operator[](Coord)`：通过传入逻辑坐标访问对应的数据元素。
* `operator()(Coord)`：通过传入逻辑坐标访问对应的数据元素。
* `operator()(Coords...)`：多参重载，等价于 `operator()(make_coord(Coords...))`。

同时，`Tensor` 支持与 `Layout` 完全同构的分层维度查询：
* `rank<I...>(Tensor)`：第 `I...` 维度的阶数（模式数）。
* `depth<I...>(Tensor)`：第 `I...` 维度的嵌套深度。
* `shape<I...>(Tensor)`：第 `I...` 维度的形状。
* `size<I...>(Tensor)`：第 `I...` 维度的容量大小。
* `layout<I...>(Tensor)`：第 `I...` 维度的内部布局。
* `tensor<I...>(Tensor)`：提取对应模式的子张量视图。

---

## 2. 存储引擎与显存空间标记 (Tensor Engines & Tagged Iterators)

### 2.1 Engine 概念
`Engine` 是对数据指针或静态数组的轻量级封装，提供了类似 `std::array` 的极简接口（包含 `iterator`、`value_type`、`reference` 及 `begin()`）。

在实际开发中，开发者无需手动构造 `Engine` 类。在调用 `make_tensor` 时，CuTe 会自动选择推导出最适合的引擎类型 —— 通常为用于寄存器数组的 `ArrayEngine<T, N>`、用于指针视图的 `ViewEngine<Iter>` 或只读视图 `ConstViewEngine<Iter>`。

---

### 2.2 显存空间强类型标记 (Tagged Iterators)

虽然任意常规裸指针都能构造 `Tensor`，但 CuTe 允许（且强烈推荐）使用显存空间标签对指针进行**强类型修饰**：

* `make_gmem_ptr(g)` 或 `make_gmem_ptr<T>(g)`：将指针 `g` 标记为**全局显存指针（Global Memory Pointer）**。
* `make_smem_ptr(s)` 或 `make_smem_ptr<T>(s)`：将指针 `s` 标记为**共享内存指针（Shared Memory Pointer）**。

> [!IMPORTANT] 为什么显存打标至关重要？
> 1. **零成本硬件指令特化**：现代 NVIDIA GPU 拥有大量专用数据搬运指令（如从全局内存直接拷贝到共享内存的 `cp.async`，或 Hopper 架构的 TMA）。这些硬件指令在微架构上严格要求“源地址在全局显存，目的地址在共享内存”。
> 2. **编译期安全审查**：通过类型打标，CuTe 的算法模板能够在编译期严格检验操作数的内存空间合法性，并在匹配时自动分发调用最高性能的 PTX 汇编指令！

---

## 3. Tensor 的创建模式：持有型 vs 非持有型 (Tensor Creation)

在 CuTe 中，张量有两种完全不同的生命周期模式：

1. **持有型张量（Owning Tensors）**：
   * 行为类似于 `std::array`。
   * 内部实际拥有存储空间（直接分配在 **GPU 寄存器** 中）。拷贝张量会触发内部元素的深拷贝；张量析构时释放空间。
2. **非持有型张量（Non-owning Tensors / Views）**：
   * 行为类似于 C++ 裸指针或 `std::span`。
   * 它仅仅是已有显存（全局内存或共享内存）的一层多维索引视图。拷贝张量仅拷贝指针与元数据，**不会发生数据深拷贝**。

> [!WARNING] 函数传参工程铁律
> 在编写通用 Tensor 函数时，**入参必须以引用或常量引用（`Tensor const&`）传递**！
> 因为若按值传递，一旦传入的是持有型寄存器张量，会导致巨大的寄存器深拷贝开销甚至寄存器溢出（Spill）。

---

### 3.1 创建非持有型视图张量 (Non-owning Tensors)

通过向 `make_tensor` 传入数据指针以及 `Layout`（或构造 Layout 的参数）来创建：

```cpp
float* A = ...;

// 1. 普通未打标指针张量
Tensor tensor_8   = make_tensor(A, make_layout(Int<8>{}));  // 静态 8 元素
Tensor tensor_8s  = make_tensor(A, Int<8>{});               // 简写，传入 Shape
Tensor tensor_8d2 = make_tensor(A, 8, 2);                   // 传入动态 Shape 与 Stride

// 2. 全局显存张量（支持混合静态/动态布局）
Tensor gmem_8s     = make_tensor(make_gmem_ptr(A), Int<8>{});
Tensor gmem_8sx16d = make_tensor(make_gmem_ptr(A), make_shape(Int<8>{}, 16));
Tensor gmem_8dx16s = make_tensor(make_gmem_ptr(A), make_shape(8, Int<16>{}),
                                                   make_stride(Int<16>{}, Int<1>{}));

// 3. 共享内存张量
Layout smem_layout = make_layout(make_shape(Int<4>{}, Int<8>{}));
__shared__ float smem[decltype(cosize(smem_layout))::value]; // 静态申请共享内存空间
Tensor smem_4x8_col = make_tensor(make_smem_ptr(smem), smem_layout);
Tensor smem_4x8_row = make_tensor(make_smem_ptr(smem), shape(smem_layout), LayoutRight{});
```

在终端对上述张量调用 `print`，输出展示了完整的类型特征：
```text
gmem_8sx16d  : gmem_ptr[32b](0x7f42efc00000) o (_8,16):(_1,_8)
smem_4x8_col : smem_ptr[32b](0x7f4316000000) o (_4,_8):(_1,_4)
```

---

### 3.2 创建持有型寄存器张量 (Owning Tensors)

持有型张量使用 `make_tensor<T>` 创建，直接映射到当前 CUDA 线程的**本地寄存器文件（Register File）**中。
由于 GPU 寄存器数量必须在编译期由 NVCC 静态确定，**持有型张量的 Shape 与 Stride 必须完全由静态编译期常量构成**！

```cpp
// 在本地寄存器中分配 4x8 的 float 张量
Tensor rmem_4x8_col = make_tensor<float>(Shape<_4, _8>{});
Tensor rmem_4x8_row = make_tensor<float>(Shape<_4, _8>{}, LayoutRight{});

// 根据已有张量的形状与步长，快速克隆出一个结构相同的寄存器张量：
Tensor rmem_4x8_like = make_tensor_like(rmem_4x8_col);
```

---

## 4. 张量元素的索引访问 (Accessing a Tensor)

通过 `operator()` 或 `operator[]` 传入逻辑坐标，Tensor 内部会自动代入 Layout 计算偏移量，并对迭代器解引用：

```cpp
Tensor A = make_tensor<float>(Shape <Shape < _4, _5>, Int<13>>{},
                              Stride<Stride<_12, _1>,     _64>{});
float* b_ptr = ...;
Tensor B = make_tensor(b_ptr, make_shape(13, 20));

// 1. 通过自然分层坐标进行读写：
for (int m0 = 0; m0 < size<0,0>(A); ++m0)
  for (int m1 = 0; m1 < size<0,1>(A); ++m1)
    for (int n = 0; n < size<1>(A); ++n)
      A[make_coord(make_coord(m0, m1), n)] = n + 2 * m0;

// 2. 通过扁平多维坐标实现矩阵转置：
for (int m = 0; m < size<0>(A); ++m)
  for (int n = 0; n < size<1>(A); ++n)
    B(n, m) = A(m, n);

// 3. 像一维线性数组一样遍历拷贝：
for (int i = 0; i < A.size(); ++i)
  A[i] = B[i];
```

---

## 5. 张量平铺与代数变换 (Tiling a Tensor)

前文在布局代数中介绍的高级操作，均可直接施加在 `Tensor` 上：

```cpp
composition(Tensor, Tiler)
logical_divide(Tensor, Tiler)
zipped_divide(Tensor, Tiler)
tiled_divide(Tensor, Tiler)
flat_divide(Tensor, Tiler)
```

通过这些代数操作，我们可以轻松从宏观大张量中提取任意分块。

> [!CAUTION] 乘法限制
> 布局乘法（`_product` 系列）**不能直接作用于 Tensor**！因为乘法会扩展值域空间（Codomain），这将导致张量去越界访问不可预知的非法内存地址。乘法仅用于构造抽象 Layout。

---

## 6. 张量切片：神奇的通配符 `_` (Slicing a Tensor)

* **传入完整坐标** $\implies$ 返回**单个标量元素**；
* **传入包含通配符 `_`（`cute::Underscore`）的切片坐标** $\implies$ 返回**降维后的局部子张量（Sub-tensor）**！

`_` 在 CuTe 中的作用完全等价于 Python/Numpy 中的 `:`：**保留该维度的全量元素，不做降维过滤**。

```cpp
// 原始形状: ((_3, 2), (2, _5, _2)):((4, 1), (_2, 13, 100))
Tensor A = make_tensor(ptr, ...);

// 切片提取列: 保留第 1 维的所有元素，第 0 维固定为 2
Tensor B = A(2, _); // ((2, _5, _2)):((_2, 13, 100))

// 切片提取行: 保留第 0 维的所有元素，第 1 维固定为 5
Tensor C = A(_, 5); // ((_3, _2)):((4, 1))

// 保持 Rank 的切片写法（使用 make_coord 显式指定嵌套切片）：
Tensor D = A(make_coord(_, _), 5); // (_3, 2):(4, 1)

// 复杂跨维度子切片提取：
Tensor E = A(make_coord(_, 1), make_coord(0, _, 1)); // (_3, _5):(4, 13)
```

切片内部执行了两项核心操作：
1. 计算部分坐标在物理内存中的基准偏移量，将底层迭代器指针推进到子张量起点；
2. 利用坐标中所有的 `_` 抽取对应的子模式，重构出全新的子布局。

---

## 7. 现代 GPU 核心分区模式 (Tensor Partitioning)

在 CUDA 编程中，张量分区（Partitioning）的核心范式是：**先平铺分块（Tiling / Divide），后切片索引（Slicing）**。业界最核心的三大分区模式如下：

### 7.1 内部分区：线程块级分块 (`inner_partition` / `local_tile`)

```cpp
Tensor A = make_tensor(ptr, make_shape(8, 24)); // 8x24 宏观全局矩阵
auto tiler = Shape<_4, _8>{};                   // 划分出 4x8 的分块

// zipped_divide 得到形状: ((_4, _8), (2, 3))
// mode-0 是局部 4x8 分块本身，mode-1 是 2x3 的分块网格
Tensor tiled_a = zipped_divide(A, tiler);

// 每个 Thread Block (CTA) 根据自己的 blockIdx 提取属于自己的 4x8 分块：
Tensor cta_a = tiled_a(make_coord(_, _), make_coord(blockIdx.x, blockIdx.y)); // (_4, _8)
```
这种模式保留内部分块模式，常被称为 **内部分区（Inner-partition）**。CuTe 将其封装为高频函数 **`local_tile`**：
```cpp
Tensor cta_a = local_tile(A, tiler, make_coord(blockIdx.x, blockIdx.y));
```

---

### 7.2 外部分区：线程级元素分发 (`outer_partition` / `local_partition`)

当一个线程块（例如包含 32 个线程）需要瓜分一个 4x8 的数据块时，每个线程提取属于自己的元素：

```cpp
// 将线程索引传入 mode-0，保留 mode-1
Tensor thr_a = tiled_a(threadIdx.x, make_coord(_, _)); // (2, 3)
```
这种模式保留外部网格，被称为 **外部分区（Outer-partition）**。高级封装函数为 **`local_partition`**。

---

### 7.3 TV 布局分区：线程-数值映射 (Thread-Value Partitioning)

这是现代高效 MMA 算子与 Tensor Core 编程的终极武器！
在 SIMT 体系下，我们需要明确定义：**哪个 Thread 负责哪几个 Value 元素？**

构建一个 `(Thread, Value)` 映射布局：

```cpp
// 构造一个 TV 布局: 将 8 个线程和每个线程持有的 4 个值，映射到 4x8 张量中
// 逻辑签名: (T8, V4) -> (M4, N8)
auto tv_layout = Layout<Shape <Shape <_2, _4>, Shape <_2,  _2>>,
                        Stride<Stride<_8, _1>, Stride<_4, _16>>>{}; // (8, 4)

// 原始数据张量 (4, 8)
Tensor A = make_tensor<float>(Shape<_4, _8>{}, LayoutRight{});

// 1. 通过复合变换，将数据在逻辑上重组为 (8 线程 x 4 数值) 的排布：
Tensor tv = composition(A, tv_layout); // (8, 4)

// 2. 每个线程用自己的 threadIdx.x 进行切片，直接提取属于自己的 4 个寄存器元素！
Tensor v = tv(threadIdx.x, _); // (4)
```

通过这一范式，任何复杂的硬件 Tensor Core 线程排布交错（Swizzle / Warp Interleaving），都被统一抽象成了纯粹的代数切片！
