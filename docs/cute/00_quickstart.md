# CuTe 快速入门指南 (Getting Started With CuTe)

> [!NOTE] 原文出处
> 本文翻译自 NVIDIA CUTLASS 官方核心文档 [`media/docs/cpp/cute/00_quickstart.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/00_quickstart.md)。

CuTe 是一套用于定义并操作**线程与数据的多层分级多维布局（Hierarchically Multidimensional Layouts）**的 C++ CUDA 模板抽象库。CuTe 提供了 `Layout`（布局）与 `Tensor`（张量）对象，紧凑地封装了数据的**类型、形状（Shape）、内存空间与排布方式**，同时为开发者自动处理极其复杂的底层索引计算。

这使得算法工程师能够专注于算法本身的**高阶逻辑描述**，而由 CuTe 负责机械、繁琐的底层索引薄记（Bookkeeping）。借助这些工具，我们能够以极高的速度设计、实现并重构各种密集线性代数算子（Dense Linear Algebra Operations）。

CuTe 最核心的抽象是**多层分级多维布局**，它可以与底层实际的数据数组进行组合从而表示张量。这种布局表达能力极其强大，几乎可以表达实现高效密集线性代数所需的一切内存排布形式。此外，布局还可以通过**函数复合（Functional Composition）**进行组合与代数变换，我们正是在此基础之上构建了平铺（Tiling）和分区（Partitioning）等一系列通用算子操作。

---

## 1. 系统要求 (System Requirements)

CuTe 遵循 CUTLASS 3.x 的软件要求：
* 支持 C++17 的主机编译器（Host Compiler）。
* 对应版本的 NVIDIA CUDA 编译器（NVCC）。

---

## 2. 前置知识储备 (Knowledge Prerequisites)

CuTe 是一个 **纯头文件（Header-only）** 的 CUDA C++ 库，要求使用 **C++17**（2017 年发布的 C++ ISO 标准）。

在通读本系列教程时，我们假定读者具备中等程度的 C++ 与 CUDA 经验：
* **C++ 经验**：能够熟练阅读和编写模板函数、模板类，理解现代 C++ 类型推导机制（如使用 `auto` 推导函数返回值类型等）。本教程在讲解时会保持循序渐进，适当解释关键的元编程细节。
* **CUDA 经验**：清晰理解设备端代码（Device Code）与主机端代码（Host Code）的本质区别、内存层级（Global Memory、Shared Memory、Register）以及如何配置并启动内核（Kernel Launch）。

---

## 3. 编译测试与示例 (Building Tests and Examples)

CuTe 的测试用例和示例程序直接作为 CUTLASS 标准构建流程的一部分运行：

* **单元测试**：位于 [`test/unit/cute`](https://github.com/NVIDIA/cutlass/tree/main/test/unit/cute) 目录下。
* **示例工程**：位于 [`examples/cute`](https://github.com/NVIDIA/cutlass/tree/main/examples/cute) 目录下。

---

## 4. 库代码结构与组织 (Library Organization)

作为一个纯头文件 C++ 库，CuTe 不需要预先编译动态库或静态库，引入头文件即可使用。库的核心头文件位于顶级目录 [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) 中，按语义划分为以下子目录：

| 子目录路径 | 包含的核心内容与功能 |
| :--- | :--- |
| [`include/cute`](https://github.com/NVIDIA/cutlass/tree/main/include/cute) | 顶层头文件对应 CuTe 最基本的构建基石，例如 [`Layout`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/layout.hpp) 与 [`Tensor`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/tensor.hpp)。 |
| [`include/cute/container`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/container) | 类似于 STL 容器的轻量级实现，如适配 CUDA 设备的 `tuple`（元组）、`array`（静态数组）与对齐数组。 |
| [`include/cute/numeric`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/numeric) | 基础数值类型支持，涵盖非标准浮点类型（如 FP8、BF16）、非标准整数、复数以及整数编译期序列。 |
| [`include/cute/algorithm`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/algorithm) | 常用实用算法实现（如 `copy` 数据拷贝、`fill` 填充、`clear` 清零），能够在可用时自动调用特定 GPU 硬件指令。 |
| [`include/cute/arch`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/arch) | 针对特定 GPU 硬件微架构的矩阵乘累加（MMA）和数据拷贝底层指令包装器（PTX 封装）。 |
| [`include/cute/atom`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/atom) | 为 `arch` 中的硬件指令提供元信息封装（Meta-information），并提供高级的分区（Partitioning）与平铺分块（Tiling）抽象。 |

---

## 5. 教程全景导读 (Tutorial Roadmap)

本教程由浅入深拆解了 CuTe 的核心体系，各章节规划如下：

* **核心必读实战**：[`0x_gemm_tutorial.md`](./0x_gemm_tutorial.md)
  * 手把手讲解如何利用 CuTe 核心组件从零构建一个密集矩阵乘法（GEMM）内核。建议初学者通读此文建立对 CuTe 的全局认知。
* **[`01_layout.md`](./01_layout.md) - Layout 核心抽象**：
  * 深入讲解 CuTe 的灵魂对象 `Layout`，解析逻辑坐标空间到物理线性索引空间的映射模型。
* **[`02_layout_algebra.md`](./02_layout_algebra.md) - 布局代数运算法则**：
  * 讲解高级 `Layout` 运算与代数系统（如 Coalesce 合并、Composition 复合、Logical Divide 逻辑划分等）。
* **[`03_tensor.md`](./03_tensor.md) - Tensor 抽象与切片**：
  * 讲解如何将 `Layout` 与物理内存指针绑定为 `Tensor`，以及多维切片与子张量视图操作。
* **[`04_algorithms.md`](./04_algorithms.md) - 泛型算法体系**：
  * 梳理操作在 `Tensor` 之上的高效泛型算法。
* **[`0t_mma_atom.md`](./0t_mma_atom.md) - MMA Atom 硬件矩阵乘抽象**：
  * 剖析 CuTe 如何将现代 NVIDIA GPU 的 Tensor Core 硬件指令（如 `mma.sync`）抽象为可泛化的 Atom 原语。
* **[`0y_predication.md`](./0y_predication.md) - 谓词判定与边缘裁剪**：
  * 解决实际工业场景中矩阵维度无法被分块整除时的越界保护与掩码机制。
* **[`0z_tma_tensors.md`](./0z_tma_tensors.md) - TMA 异步硬件张量拷贝**：
  * 针对 Hopper (H100) 及 Blackwell 架构的硬件级张量内存加速器（TMA）的高级 Tensor 支持。

---

## 6. 核心调试技巧与打印机制 (Quick Tips)

在 GPU Kernel 这种高度并发、不可见内部状态的环境中，**如何打印和调试 CuTe 对象**是至关重要的技能。

### 6.1 主机端与设备端的统一打印

`cute::print` 函数为几乎所有 CuTe 内置类型提供了重载，包括指针（Pointers）、整数、步长（Strides）、形状（Shapes）、布局（Layouts）和张量（Tensors）。
当你对某个对象的内部状态不确定时，直接调用 `print` 即可。

> [!WARNING] 设备端打印开销陷阱
> `cute::print` 可以在主机端（Host）或设备端（Device）无缝调用。但请务必注意：**在 GPU 设备端打印非常昂贵**！
> 即使你写在了一个实际上永远不会执行的分支里（如 `if (false) { print(...); }`），或者从未触发的条件分支中，设备端打印机制依然可能导致 NVCC 编译器生成更加保守、性能更差的汇编指令。
> **在完成内核调试后，务必将设备端所有打印代码彻底删除或使用宏开关屏蔽！**

### 6.2 仅在特定线程打印

为了避免几千个线程同时在终端疯狂刷屏，标准工程实践是**仅允许全局第 0 号线程**打印：

`cute::thread0()` 函数在当前处于整个 Kernel Grid 的全局第 0 号线程（即 Block 0 的 Thread 0）时返回 `true`。这是最常用的调试范式：

```cpp
if (cute::thread0()) {
  cute::print(some_cute_object);
}
```

如果某些特定算法逻辑运行在特定线程块或线程中，你可以引入头文件 [`cute/util/debug.hpp`](https://github.com/NVIDIA/cutlass/tree/main/include/cute/util/debug.hpp)，使用高级过滤函数：

```cpp
// 仅在 threadIdx == tid 且 blockIdx == bid 时执行
if (cute::thread(tid, bid)) {
  cute::print(my_local_tensor);
}
```

### 6.3 丰富的格式化可视化函数

除了普通的 `cute::print`，CuTe 还提供了几种针对特定维度的强力可视化函数：

1. **`cute::print_layout`**：
   * 将任意 Rank-2（2 维）的布局以纯文本 ASCII 表格的形式渲染出来。这是直观观察从“多维坐标到一维物理索引映射”的神器。
2. **`cute::print_tensor`**：
   * 支持打印 Rank-1 到 Rank-4 的张量数据，并在控制台以多维表格形式排列其实际存储的数值。常用于验证数据搬运（Copy）后分块数据是否符合预期。
3. **`cute::print_latex`**：
   * **杀手级功能**：生成标准 LaTeX 源码，可通过 `pdflatex` 编译成带有彩色高亮单元格的高清 PDF 图表！它原生支持 `Layout`、`TiledCopy` 和 `TiledMMA`，能够极其清晰地展现多线程之间数据分配与切片分布模式。
