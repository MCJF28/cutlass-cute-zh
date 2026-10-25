# 09. Copy Atom 硬件搬运抽象 (UniversalCopy, cp.async & ldmatrix)

> **重要免责声明**：本网站只是用来消耗反重力额度的，并且给有需要的人看一看，准确度极低。可用度极低，不想看的请不看，造成错误概不负责。
> 本篇由 AI 辅助汉化与深度解析，非 NVIDIA 官方技术文档。涉及 GPU 底层硬件架构参数与生产算子调优，请以 [NVIDIA CUTLASS 官方源码](https://github.com/NVIDIA/cutlass) 为准。

---

## 1. 为什么数据搬运需要硬件级抽象？

在 GPU 高性能计算中，**算力与访存带宽之间存在严重的失衡**。为了跑满 Tensor Core 惊人的吞吐量，我们必须以极致的硬件效率将数据从片外全局内存（GMEM）搬运到片上共享内存（SMEM），再从共享内存加载到寄存器堆（RF）。

在传统 CUDA 编程中，开发者通常需要手写大量的内联 PTX 汇编与繁琐的多维线程下标换算：
- 全局内存到共享内存：Ampere 引入的 `cp.async.cg` 异步硬件拷贝指令；
- 共享内存到寄存器：针对 Tensor Core 排布优化的 `ldmatrix.sync.aligned.m8n8.x4` 指令；
- 寄存器到共享内存：`stmatrix` 专有转置写回指令。

这些底层硬件指令对**参与线程数、数据对齐边界（128-bit）、每线程读取字节数、寄存器通道映射**有着极为苛刻的物理约束。一旦手动推导出错，轻则产生非对齐陷阱（Illegal Address），重则数据被错位混淆。

CuTe 的 **Copy Atom 体系**正是为了将这些苛刻的硬件原语封装为代数化的泛型组件，使我们可以像拼积木一样组合出工业级的极致访存流水线。

---

## 2. Copy Atom 的四层核心代数模型

与计算端的 MMA Atom 镜像对称，CuTe 的数据搬运抽象同样由四层结构逐级放大：

```
+--------------------------------------------------------------+
| 4. copy(tiled_copy, tS, tD)       (泛型算法调用)             |
+--------------------------------------------------------------+
| 3. TiledCopy                      (分块平铺：宏观数据搬运器) |
+--------------------------------------------------------------+
| 2. Copy_Traits                    (指令元数据与 TV 布局约束) |
+--------------------------------------------------------------+
| 1. Copy Operation / UniversalCopy (最底层 PTX 指令或标量读写)|
+--------------------------------------------------------------+
```

### 2.1 底层搬运原语（Copy Operation）

CuTe 预置了覆盖各代 NVIDIA GPU 的硬件指令封装，定义在 `cute/arch/copy_*.hpp` 中：
- `UniversalCopy<uint128_t>`：通用的 128 位宽拷贝（当无法匹配专用 PTX 时退化为向量加载/存储）；
- `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`：Ampere 架构的 `cp.async.ca` 异步拷贝，直接绕过寄存器将 GMEM 搬至 SMEM；
- `SM75_U32x4_LDSM_N`：Turing/Ampere 架构的 `ldmatrix.x4`，单条指令由一个 Warp（32 线程）协作，从 SMEM 中抓取 4 个 $8 \times 8$ 的矩阵片到寄存器。

### 2.2 Copy Atom (`Copy_Atom`)

`Copy_Atom<CopyOp, ElementType>` 是指令的代数载体。它明确定义了：
1. **源布局与目标布局**：单次指令调用消费的数据形态；
2. **Thread-Value (TV) 映射**：参与该指令的线程编号与每个线程负责搬运的元素下标。

例如，对于 `SM75_U32x4_LDSM_N`：
- 线程数为 32（一个 Warp）；
- 每个线程读取 16 字节（4 个 32 位寄存器）；
- 硬件执行时自动完成转置或直通重排，以匹配 Tensor Core 的输入需求。

---

## 3. 分块平铺：`make_tiled_copy`

单条硬件指令所能搬运的数据量通常很小（如 16 字节），而一个线程块（CTA）在每个流水线阶段需要搬运数十 KB 的巨型矩阵块（例如 $128 \times 64$ 个 `half` 元素）。
`make_tiled_copy` 的任务，就是**将一个细粒度的 Copy Atom 在空间上平铺成一个满足整块需求的宏观搬运器**。

### 3.1 `make_tiled_copy` 的核心参数

```cpp
template <class CopyAtom,
          class ThreadLayout,
          class ValueLayout = Layout<Shape<_1>>>
auto make_tiled_copy(CopyAtom const& copy_atom,
                     ThreadLayout const& thr_layout,
                     ValueLayout const& val_layout = {});
```

- **`CopyAtom`**：基础硬件搬运原子（例如 `Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half>`）；
- **`ThreadLayout`**：分配执行该搬运器的线程排布形状（例如 `Shape<_32, _4>{}` 表示 128 个线程以 $32 \times 4$ 网格排布）；
- **`ValueLayout`**：每个线程在单次原子指令内搬运的值排布。通常由 Copy Atom 的位宽自动推导（例如 128 位对应 8 个 `half`，即 `Shape<_8>{}`）。

### 3.2 向量化对齐与步长折叠

当我们将一个 128 位的 `cp.async` 原子平铺在 $32 \times 4$ 的线程网格上时：
- 第 0 维（32 个线程）每人搬运 8 个连续元素，总共覆盖 $32 \times 8 = 256$ 字节；
- 第 1 维（4 个线程）覆盖 4 个不同的行；
- CuTe 的编译器会自动验证每一行在物理内存中的对齐是否满足 16 字节边界，杜绝任何运行期硬件异常。

---

## 4. 线程切片与张量分区 (Partition)

构造出 `TiledCopy` 后，在 GPU 内核中每个线程如何获取属于自己的那部分数据？CuTe 提供了极其优雅的切片 API。

### 4.1 获取线程切片 (`get_thread_slice`)

```cpp
auto tiled_copy = make_tiled_copy(...);

// 每个物理线程提取专属于自己的逻辑切片视图
auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
```

### 4.2 源与目标分区：`partition_S` 与 `partition_D`

假设我们有一个全局内存张量 `gA`（源，Source）和一个共享内存张量 `sA`（目标，Destination）：

```cpp
// 对源张量进行分区，提取该线程负责读取的坐标子集
Tensor tAgA = thr_copy.partition_S(gA); 
// 形状展开为: ((CPY_V, CPY_M, CPY_K), Tile_M, Tile_K)

// 对目标张量进行分区，提取该线程负责写入的坐标子集
Tensor tAsA = thr_copy.partition_D(sA);
// 形状展开为完全对称的布局
```

此时，`tAgA` 与 `tAsA` 的形状与步长严格同构！线程不再需要关心全局矩阵的行列坐标换算，只需直接调用泛型算法：

```cpp
// 发射硬件搬运指令
cute::copy(tiled_copy, tAgA, tAsA);
```

这一行看似简单的 `cute::copy`，在编译期会被展开为多条连续的、具备最高吞吐的 `cp.async` 硬件指令！

---

## 5. 完整实战范式：异步 128-bit 流水线数据加载内核

```cpp
#include <cute/tensor.hpp>
using namespace cute;

template <class Element, class SmemLayout>
__global__ void async_load_kernel(const Element* gmem_ptr, SmemLayout smem_layout) {
  extern __shared__ char raw_smem[];
  Element* s_ptr = reinterpret_cast<Element*>(raw_smem);

  // 1. 包装 GMEM 与 SMEM 张量 (128x64 矩阵分块)
  Tensor gA = make_tensor(make_gmem_ptr(gmem_ptr), make_shape(Int<128>{}, Int<64>{}), LayoutRight{});
  Tensor sA = make_tensor(make_smem_ptr(s_ptr), smem_layout);

  // 2. 构造 128-bit 异步硬件拷贝原子 (8 个 half = 128 bit)
  using CopyAtomType = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>;
  
  // 3. 构造平铺搬运器：128 线程 (16 x 8 排布)
  auto tiled_copy = make_tiled_copy(
      CopyAtomType{},
      make_layout(make_shape(Int<16>{}, Int<8>{}), LayoutRight{}),
      make_layout(make_shape(Int<1>{}, Int<8>{})) // 每个线程在列方向连续读 8 个元素
  );

  // 4. 获取线程切片并分区
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
  Tensor tAgA = thr_copy.partition_S(gA);
  Tensor tAsA = thr_copy.partition_D(sA);

  // 5. 提交异步拷贝
  cute::copy(tiled_copy, tAgA, tAsA);

  // 6. 提交 Ampere 硬件异步组提交与等待
  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // 此时 sA 中已全带宽填满数据，可供 Tensor Core 消费！
}
```

---

## 6. 核心认知与进阶指南

1. **`partition_S` 与 `partition_D` 的不变性保证**：无论外部张量是连续的、带 Swizzle 的还是非连续跨步的，分区后的张量在最内层都保持与 Copy Atom 的值布局强类型对齐；
2. **结合 `copy_if` 处理边界**：当矩阵尺寸不能被 Tile 整除时，`tiled_copy` 可以与谓词张量结合，无缝转化为 `copy_if`，硬件层自动为越界地址补 0，无需分支判断。
