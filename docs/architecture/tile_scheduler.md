# 18. 持久化内核与工作瓦片调度器 (Tile Scheduler)

> **重要免责声明**：本网站只是用来消耗反重力额度的，并且给有需要的人看一看，准确度极低。可用度极低，不想看的请不看，造成错误概不负责。
> 本篇由 AI 辅助汉化与深度解析，非 NVIDIA 官方技术文档。涉及 GPU 底层硬件架构参数与生产算子调优，请以 [NVIDIA CUTLASS 官方源码](https://github.com/NVIDIA/cutlass) 为准。

---

## 1. 传统 GPU Grid 调度的瓶颈与“尾部效应”

在传统 CUDA 编程范式中，执行一个大矩阵乘法通常通过启动一个庞大的三维网格（Grid）来完成：
```cpp
dim3 gridDim(ceil(M / TileM), ceil(N / TileN), Batch);
dim3 blockDim(128);
gemm_kernel<<<gridDim, blockDim>>>(...);
```
硬件芯片内部的 **GigaThread 硬件调度器** 负责将数以万计的 CTA（线程块）动态轮询分发给物理 SM。然而，在现代超大规模 GPU（如拥有 132 个 SM 的 Hopper H100 或拥有 160+ SM 的 Blackwell B200）上，这种粗放式的传统调度面临着三大性能陷阱：

### 1.1 尾部效应（Tail Effect / Wave Quantization）

假设 GPU 拥有 132 个 SM，每个 SM 同一时刻只能驻留 1 个全尺寸 CTA。
- 若总计算任务切分为 133 个 Tile，前 132 个 Tile 会在第 1 波（Wave 1）中被 132 个 SM 并发算完；
- 剩下的 **仅有 1 个 Tile** 必须在第 2 波（Wave 2）中单独占用 1 个 SM 运行，而其余 **131 个 SM 完全闲置处于死等状态**！
- 此时整个内核的执行时间翻倍，GPU 总体有效利用率瞬间暴跌近 50%！

```
Wave 1: [SM 0] [SM 1] [SM 2] ... [SM 131]  -> 132 个 SM 全部打满 (利用率 100%)
Wave 2: [SM 0] [ 空闲 ] [ 空闲 ] ... [ 空闲 ]  -> 仅 1 个 SM 工作 (利用率 0.75%!)
```

### 1.2 缓存局部性（Cache Locality）破坏

硬件 GigaThread 调度器主要关注负载均衡，并不深度感知矩阵 $A$ 与 $B$ 的数据局部性。若相邻被调度的 CTA 访问了跨度巨大的内存地址，会导致片上数十 MB 的二级缓存（L2 Cache）频繁发生抖动换出，有效访存带宽大打折扣。

---

## 2. 持久化内核（Persistent Kernel）范式

为了攻克上述难题，CUTLASS 3.x 全面转向了**持久化内核（Persistent Kernel）与软件可编程瓦片调度器（Tile Scheduler）**架构！

### 2.1 什么是持久化内核？

- **固定网格尺寸**：内核启动时不再依据矩阵大小计算 Grid，而是仅启动**恰好能够填满整张 GPU 物理 SM 的常驻 CTA 数量**（例如针对 H100 启动 132 个 CTA）；
- **工作池轮询（Work-stealing / Loop）**：每个常驻 CTA 进入一个持久化 `while` 循环，向调度器动态“领任务（Fetch Next Work）”；
- **就地常驻**：计算完当前 Tile 后，CTA 并不销毁退出，而是直接在当前 SM 上复用共享内存与寄存器堆，立即领取下一个 Tile 进行计算，直到调度器宣布所有 Tile 处理完毕后统一切出。

```cpp
// 持久化内核内部的标准化主循环骨架
TileScheduler scheduler(params.scheduler);
auto work_tile_info = scheduler.get_initial_work();

while (work_tile_info.is_valid()) {
  // 1. 执行当前 Tile 的 GEMM 主循环计算
  collective_mainloop.load_and_compute(work_tile_info, ...);

  // 2. 尾声处理与写回
  collective_epilogue.store(work_tile_info, ...);

  // 3. 向调度器请求下一个工作瓦片
  work_tile_info = scheduler.fetch_next_work();
}
```

---

## 3. CUTLASS 3.x 核心调度策略深度剖析

CUTLASS 3.x 提供了多种针对不同工作负载量身定制的调度器实现，均位于 `cutlass::gemm::kernel::` 命名空间下。

### 3.1 静态光栅化调度器（Raster-Order Scheduler）

这是最通用的基础调度策略。它在软件层面重新规划了 CTA 遍历矩阵瓦片的空间几何路径，从而最大化 L2 Cache 命中率。

- **蛇形遍历（Snake Walk / Hilbert Curve）**：
  在常规的行优先扫描中，当一行结束跳到下一行开头时，矩阵 $B$ 的缓存早已全部失效。而蛇形遍历在偶数行从左往右扫描，在奇数行从右往左扫描，使转折处的 CTA 能够立即无缝复用上一行末尾留在 L2 Cache 中的矩阵 $B$ 缓存！
- **最大驻留深度（Max Swizzle Size）**：
  通过 `raster_order::AlongM` 或 `raster_order::AlongN` 参数，将连续的多个 CTA 聚束在同一个局部分块内，确保共享的权重矩阵 Tile 在 L2 Cache 内驻留时间最大化。

### 3.2 Stream-K 动态负载均衡调度器

**Stream-K 是 CUTLASS 近年来在体系结构领域的里程碑式突破！**
它的核心洞察是：**摆脱以矩阵 Tile 为离散原子单位的传统观念，将整个 GEMM 的全部乘加累加（MAC）计算量视为一条连续的数据流（Stream of MACs）**。

1. **绝对均分**：将总计算量严格除以可用的物理 SM 数量（例如 132 等分）；
2. **跨界 Tile 协作（Split-K within Stream）**：
   - 如果某个工作分块恰好处在两个 SM 的分配交界处，Stream-K 允许第一个 SM 计算该 Tile 的前 60% 累加，第二个 SM 计算后 40% 累加；
   - 两个 SM 通过原子锁与片上 L2 同步完成归约；
3. **彻底消灭尾部效应**：所有 SM 的工作量在纳秒级精度上绝对对齐，所有 SM 同时开始、同时结束，GPU 占用率无论矩阵尺寸奇偶均稳稳达到 **99.9% 理论极值**！

### 3.3 组分块调度器（Grouped GEMM Scheduler）

针对大语言模型中混合专家系统（MoE, Mixture of Experts）的稀疏路由场景：
- 输入为动态不均衡的多个子矩阵（各个 Expert 分到的 Token 数量各不相同）；
- 调度器维护一个全局元数据队列，自动将不同大小的 Expert GEMM 任务压平并动态负载分发给常驻 CTA，消除因某个 Expert 数据过长而导致其余 SM 饥饿的现象。

---

## 4. 调度器的编译期与运行期协同

在 CUTLASS 3.x 中，调度器也是通过模板策略注入到 `GemmUniversalAdapter` 中的：

```cpp
#include <cutlass/gemm/kernel/tile_scheduler.hpp>

// 选用支持 TMA 与持久化网格的静态调度器
using TileScheduler = cutlass::gemm::kernel::PersistentTileSchedulerSm90;

// 构造调度器运行期参数
typename TileScheduler::Params scheduler_params = 
    TileScheduler::to_underlying_arguments({
        problem_shape_mnkl,
        tile_shape_mnk,
        cluster_shape_mnk
    });
```

在内核执行过程中，调度器内部使用了紧凑的无锁位运算（Lock-free Bitops）与轻量原子计数器，每个 CTA 获取下一个 Tile 的延迟仅需几个时钟周期，相比传统内核反复启动（Kernel Launch）的数微秒开销，性能提升了数个数量级！

---

## 5. 总结与调优建议

1. **超大矩阵（Full Wave）**：优先使用 `PersistentTileSchedulerSm90` 配合蛇形遍历，充分压榨 L2 Cache 带宽；
2. **非规则矩阵与长条薄矩阵（Irregular / Tail Heavy）**：果断选用 `StreamK` 调度器，彻底抹平尾波延迟；
3. **MoE 稀疏专家层**：必须配合 `GroupedGemmTileScheduler`，以动态负载抢占模式化解路由倾斜（Router Skew）。
