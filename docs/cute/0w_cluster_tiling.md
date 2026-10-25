# 13. 线程块集群协同平铺 (Cluster Tiling & DSMEM)

> **重要免责声明**：本网站只是用来消耗反重力额度的，并且给有需要的人看一看，准确度极低。可用度极低，不想看的请不看，造成错误概不负责。
> 本篇由 AI 辅助汉化与深度解析，非 NVIDIA 官方技术文档。涉及 GPU 底层硬件架构参数与生产算子调优，请以 [NVIDIA CUTLASS 官方源码](https://github.com/NVIDIA/cutlass) 为准。

---

## 1. 硬件革命：Hopper / Blackwell 线程块集群与 DSMEM

在 NVIDIA Ampere 架构及以前，GPU 的计算并发模型由两级构成：
1. **线程级（Thread / Warp）**：由同一个 SM 内调度执行；
2. **线程块级（Thread Block / CTA）**：各自独立映射到不同的 SM，彼此之间完全隔离，唯一的通信途径是通过高延迟、高功耗的片外 DRAM 或二级缓存（L2 Cache）。

然而，随着大模型注意力机制与超大矩阵乘法的爆发，L2 Cache 逐渐成为带宽瓶颈。为此，NVIDIA 在 **Hopper（SM90）与 Blackwell（SM100）** 架构中引入了革命性的硬件新层级：**线程块集群（Thread Block Cluster）**。

```
+-------------------------------------------------------------+
|              Thread Block Cluster (例如 2x2 CTA)             |
|                                                             |
|  +-----------------------+       +-----------------------+  |
|  | CTA (0, 0) on SM 0    |  <->  | CTA (0, 1) on SM 1    |  |
|  | Local SMEM A          |  片上 | Local SMEM B          |  |
|  +-----------------------+  NoC  +-----------------------+  |
|              ^                           ^                  |
|              |         高速互联          |                  |
|              v                           v                  |
|  +-----------------------+       +-----------------------+  |
|  | CTA (1, 0) on SM 2    |  <->  | CTA (1, 1) on SM 3    |  |
|  | Local SMEM C          |  互访 | Local SMEM D          |  |
|  +-----------------------+       +-----------------------+  |
+-------------------------------------------------------------+
                               |
                        Distributed SMEM
```

### 1.1 核心硬件新特性
- **分布式共享内存（Distributed Shared Memory, DSMEM）**：同一个 Cluster 内的所有 CTA 的物理共享内存通过片上高速互连网络（NoC）打通为一个统一的寻址空间。CTA 0 可以直接通过专用 PTX 指令或指针解引用，直接读写 CTA 1 的共享内存，延迟仅为 L2 Cache 访问的几分之一！
- **TMA 硬件多播（TMA Multicast）**：TMA 引擎在从全局内存搬运数据时，可以根据掩码（Multicast Mask）一次性将同一块数据**直接广播推送到整个 Cluster 内的多个 CTA 共享内存中**，而全局内存仅需读取一次，节省高达 $2\times \sim 8\times$ 的片外访存带宽！
- **Cluster 级硬件同步屏障**：无需退出内核即可在 Cluster 内部进行超轻量级硬件栅障同步。

---

## 2. CuTe 中的 Cluster Tiling 抽象

传统的 CuTe Tiling 通常是在 CTA 局部坐标系下进行分块：
$$\text{CTA Tile} = (\text{BLK\_M}, \text{BLK\_N}, \text{BLK\_K})$$

当引入 Cluster 后，计算网格的平铺空间被扩展为一个**两级分层张量**：
$$\text{Cluster Shape} = (\text{C\_M}, \text{C\_N})$$
$$\text{Grid Shape} = ((\text{BLK\_M}, \text{BLK\_N}), (\text{C\_M}, \text{C\_N}), (\text{REST\_M}, \text{REST\_N}))$$

CuTe 通过精妙的层次化布局（Hierarchical Layout），将 Cluster 几何拓扑编码进类型系统，使得编译器在生成指令时能精准知道：
1. 当前 CTA 处于 Cluster 的哪个逻辑坐标；
2. 哪个数据块应当本地处理，哪个数据块应当通过 TMA Multicast 广播。

---

## 3. TMA Multicast 广播代数与掩码计算

在大矩阵乘法 $C = A \times B$ 中：
- 矩阵 $A$ 的一个 Tile 需要被沿 $N$ 方向的所有 CTA 共享；
- 矩阵 $B$ 的一个 Tile 需要被沿 $M$ 方向的所有 CTA 共享。

在由 $2 \times 2$ 个 CTA 构成的 Cluster 中：
- 当搬运 $A$ 的分块时，`CTA(0, 0)` 与 `CTA(0, 1)` 需要接收完全相同的数据；
- 当搬运 $B$ 的分块时，`CTA(0, 0)` 与 `CTA(1, 0)` 需要接收完全相同的数据。

### 3.1 组装多播掩码（Multicast Mask）

在 PTX 硬件层面，TMA Multicast 指令接收一个 16-bit 整数作为多播目标 CTA 的位掩码：

```cpp
// 假设 Cluster 形状为 2x2:
// CTA 0 位于 (0, 0), CTA 1 位于 (1, 0), CTA 2 位于 (0, 1), CTA 3 位于 (1, 1)

// 若广播给第 0 行的所有 CTA (CTA 0 和 CTA 2):
uint16_t mcast_mask_a = (1 << 0) | (1 << 2); // 0b0101

// 若广播给第 0 列的所有 CTA (CTA 0 和 CTA 1):
uint16_t mcast_mask_b = (1 << 0) | (1 << 1); // 0b0011
```

### 3.2 CuTe 自动化 Multicast 编排

CuTe 在 `make_tma_copy` 中原生支持多播描述符绑定：

```cpp
#include <cute/tensor.hpp>
using namespace cute;

// 定义 2x2 Cluster 几何拓扑
using ClusterShape = Shape<Int<2>, Int<2>, Int<1>>;

// 针对张量 A 沿 Cluster 的第二维 (N 维) 进行组播
auto tma_load_a = make_tma_copy_multicast(
    Copy_Atom<SM90_U32x4_STSM_N, ElementA>{},
    gA,                      // 全局内存张量
    sA_layout,               // 共享内存布局
    ClusterShape{},          // 集群形状
    Int<1>{}                 // 多播轴 (Axis 1 即 N 维度)
);
```

通过这一声明，CuTe 会在编译期自动推导出硬件 TMA Descriptor 所需的所有空间步长与多播位掩码，并在指令发射时由硬件自主广播，内核中无需编写任何分支判断代码！

---

## 4. 分布式共享内存指针：`make_smem_ptr` 与跨 SM 寻址

当需要直接访问 Cluster 内邻居 SM 的共享内存时，CuTe 提供了寻址空间泛化能力：

### 4.1 DSMEM 指令与物理映射

在 PTX 汇编中，访问邻居 CTA 共享内存使用 `mapa.shared::cluster.u32` 指令，该指令输入一个当前 CTA 的共享内存偏移量以及目标 CTA 的 Rank 编号，返回一个在片上 NoC 范围内的全局唯一寻址指针。

CuTe 将其封装为强类型指针修饰符：

```cpp
// 获取当前 CTA 在 Cluster 中的相对 rank
int cluster_rank = cute::block_rank_in_cluster();

// 将本地共享内存指针转换为目标 CTA 的跨 SM 远程指针
auto remote_ptr = cute::cast_smem_ptr_to_cluster(local_smem_ptr, target_rank);

// 包装为远程张量视图
auto remote_tensor = make_tensor(remote_ptr, sA_layout);

// 像操作本地内存一样读取目标 SM 的张量数据：
// 硬件片上 NoC 自动路由，无需经过片外 DRAM！
auto val = remote_tensor(0, 0);
```

---

## 5. 性能收益与工业级实战意义

在 H100 SXM5 / B200 平台上运行大型 GEMM（如 $M=4096, N=4096, K=4096$）时：
1. **DRAM 带宽消耗降低 $50\% \sim 75\%$**：通过 $2 \times 2$ 或 $1 \times 4$ Cluster 多播，全局内存加载次数直接折半甚至降为原本的四分之一；
2. **多阶段流水线泡泡（Pipeline Bubble）大幅缩减**：数据提前在 DSMEM 中就绪，计算单元几乎 100% 保持在饱和计算状态；
3. **Blackwell 架构 2-SM 协同的物理基石**：Blackwell 引入的 `tcgen05` 双 SM 协同执行正是建立在 Cluster 与 DSMEM 之上的极致进化形态！
