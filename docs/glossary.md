# CUTLASS & CuTe 核心术语与代数符号对照表 (Glossary)

为了保证整套文档在理论严谨性与中文工程阅读习惯上的高度统一，本项目建立并严格遵循以下专业术语规范。本表涵盖 CuTe 布局代数、CUDA 硬件层次、分块抽象与软件流水线。

---

## 1. 核心布局代数与概念 (Layout Algebra)

| 英文原词 (English) | 标准中文译名 | 定义与代数含义说明 |
| :--- | :--- | :--- |
| **Layout** | **布局** | 一个从逻辑多维坐标空间（Coordinate Space）映射到一维物理索引空间（Index Space）的函数 $f: \mathcal{C} \to \mathcal{I}$。由 `Shape` 和 `Stride` 组成。 |
| **Shape** | **形状** | 逻辑坐标空间的范围（域的大小）。可以是整数标量，也可以是嵌套元组（Nested Tuple）。 |
| **Stride** | **步长** | 沿各个逻辑坐标维度移动 1 个单位时，物理内存索引增加的偏移量。 |
| **Coordinate Space** | **坐标空间** | 程序员在算法逻辑上感知到的坐标系（例如矩阵的第 $i$ 行、第 $j$ 列）。 |
| **Index Space** | **索引空间** | 物理内存或连续寄存器的一维线性偏移量（如连续字节或元素下标）。 |
| **Hierarchical Layout** | **分层/嵌套布局** | `Shape` 或 `Stride` 中包含元组（如 `(Shape_A, (Shape_B, Shape_C))`）的复杂布局，用于表达张量在 Warp、Thread、Iteration 级别的多层分块。 |
| **Layout Algebra** | **布局代数** | 对 Layout 进行代数运算（合并、复合、划分等）的整套数学体系。 |
| **Coalesce** | **合并化简** | 将连续且步长相容的相邻维度合并为一个更简洁维度的代数操作。 |
| **Composition** | **复合 (运算)** | 函数复合运算 $A \circ B$。先应用布局 $B$ 的坐标映射，再代入布局 $A$ 中。在代码中表示为 `composition(A, B)`。 |
| **Complement** | **补全 (运算)** | 给定一个布局 $A$ 和目标形状大小，计算出一个不与 $A$ 覆盖的索引重叠、且恰好填补剩余空间的布局。常用于消解跨线程数据依赖。 |
| **Logical Division** | **逻辑划分** | 用一个子布局去划分父布局，得到多级嵌套的逻辑网格结构。代码为 `logical_divide(A, B)`。 |
| **Tiled Division** | **平铺/分块划分** | 将一个大布局切分为若干固定大小的 Tile（分块）。 |
| **Congruent** | **全等 / 同构** | 两个布局的层次结构、形状与步长在代数上完全一一对应。 |
| **Compatible** | **相容** | 一个坐标或形状可以无歧义地作为输入代入某个布局中进行计算。 |

---

## 2. 张量与内存抽象 (Tensor & Memory)

| 英文原词 (English) | 标准中文译名 | 定义与说明 |
| :--- | :--- | :--- |
| **Tensor** | **张量** | 数据指针引擎（Engine）与布局（Layout）的绑定体。`Tensor = Engine + Layout`。 |
| **Engine** | **存储引擎 / 指针** | 实际指向物理内存（全局内存、共享内存、寄存器）的迭代器或裸指针包装。 |
| **Sub-tensor** | **子张量** | 对父张量执行切片（Slice）或局部索引后得到的较小张量视图（零拷贝引用）。 |
| **Slice** | **切片** | 使用特定坐标或通配符 `_` 取出张量某个维度截面的操作。 |
| **Swizzle** | **地址混洗** | 通过异或（XOR）等位运算重新排列内存地址，彻底消除多线程访问共享内存时的 **Bank Conflict（存储体冲突）**。 |
| **Bank Conflict** | **存储体冲突** | 多个线程同时访问同一个共享内存 Bank 的不同地址导致的串行化开销。 |

---

## 3. 硬件映射与算子原子 (Hardware Mapping & Atoms)

| 英文原词 (English) | 标准中文译名 | 定义与说明 |
| :--- | :--- | :--- |
| **MMA (Matrix Multiply-Accumulate)** | **矩阵乘累加** | 现代 GPU 硬件执行 $D = A \times B + C$ 的核心计算单元。 |
| **MMA Atom** | **MMA 原语 / 硬件原子** | 单条硬件指令（如 `mma.sync`、`wgmma`）能完成的最小矩阵乘单元（如 $16 \times 8 \times 16$）。 |
| **TiledMMA** | **分块 MMA** | 将 MMA Atom 沿 $M, N, K$ 维度在 Thread Block 或 Warp 级别平铺扩展后的大尺寸矩阵乘执行器。 |
| **TiledCopy** | **分块拷贝** | 将一条硬件数据载入指令（如 `cp.async`、`ldmatrix`）扩展至多线程协同读取的大块数据搬运器。 |
| **ThrValPartition** | **线程-数值分区** | 将一个逻辑张量划分并分配给具体线程寄存器的映射（定义哪个线程持有哪几个数据元素）。 |
| **WGMMA (Warpgroup MMA)** | **Warp 组矩阵乘** | NVIDIA Hopper (H100) 及更新架构引入的，由 4 个 Warp（128 个线程）联合驱动异步 Tensor Core 计算的高阶指令。 |
| **TMA (Tensor Memory Accelerator)** | **张量内存加速器** | NVIDIA Hopper / Blackwell 引入的独立硬件单元，支持从全局内存到共享内存的多维张量异步硬件搬运（无需占用 CUDA Core）。 |

---

## 4. 软件流水线与调度 (Pipeline & Scheduling)

| 英文原词 (English) | 标准中文译名 | 说明 |
| :--- | :--- | :--- |
| **Software Pipeline** | **软件流水线** | 隐藏内存访问延迟的核心技术，将数据加载（Load）与计算（MMA）重叠执行。 |
| **Multi-stage Pipeline** | **多级流水线** | 采用三级或四级缓冲（Double/Triple/Quad Buffering）构建的异步流水阶段。 |
| **Predication** | **谓词判定 / 掩码判断** | 在边界非对齐矩阵乘中，通过布尔条件（Predicate）禁止无效线程读写非法越界内存。 |
| **Prologue / Mainloop / Epilogue** | **序言段 / 主循环段 / 尾声段** | 高性能 GEMM 的标准三段式结构：预取数据 $\to$ 循环计算 $\to$ 写回输出并应用激活函数。 |

---

> [!TIP]
> 读者在阅读全套文档过程中，可随时对照本术语表。若在不同页面遇到相同的代数运算符号，请严格以本表的数学定义为准。
