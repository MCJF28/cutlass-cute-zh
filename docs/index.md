---
layout: home

hero:
  name: "CUTLASS 3.x & CuTe"
  text: "底层高性能 GPU 算子开发与核心架构解析"
  tagline: "全网首部系统化超详尽架构专著（AI 辅助深度汉化、数学推导与源码剖析版）"
  actions:
    - theme: brand
      text: 快速开始 (Quickstart)
      link: /cute/00_quickstart
    - theme: alt
      text: 权威术语词典 (Glossary)
      link: /glossary
    - theme: alt
      text: 从零手写 GEMM 教程
      link: /cute/0x_gemm_tutorial

features:
  - icon:
      svg: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>'
    title: 严谨布局代数 (Layout Algebra)
    details: 彻底拆解 Shape、Stride、分层嵌套布局、Composition、Coalesce 与 Complement，将高维张量坐标精准映射至物理内存。
  - icon:
      svg: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg>'
    title: 硬件级 MMA 与 TMA 抽象
    details: 深入 Hopper 与 Blackwell 架构的 WGMMA (Warpgroup MMA)、Tensor Memory Accelerator (TMA) 异步硬件加速通道。
  - icon:
      svg: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>'
    title: 现代算子核心基石
    details: FlashAttention-2/3、大模型 FP8/FP4 混合精度量化、DeepSeek 极致算子优化的底层支撑核心。
  - icon:
      svg: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/><path d="M6 6h10"/><path d="M6 10h10"/></svg>'
    title: 万字长篇架构专著
    details: 30+ 篇深度专著，每章配备万字级别的深入推导、内存布局 ASCII 可视化、PTX 汇编映射与工业级避坑指南。
  - icon:
      svg: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 21h5v-5"/></svg>'
    title: 异步软件流水线
    details: 深入剖析生产者-消费者解耦模型、Warp Specialization、硬件事务屏障 (mbarrier) 与相位反转无锁同步。
  - icon:
      svg: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>'
    title: 拥抱 Blackwell 前沿
    details: 首发第 5 代 Tensor Core tcgen05 指令集、OCP MXFP4 / NVFP4 块级微缩放 (Microscaling) 与双 SM 协同执行机制。
---

::: warning 重要免责与特别提示
**本网站只是用来消耗反重力额度的，并且给有需要的人看一看，准确度极低。可用度极低，不想看的请不看，造成错误概不负责。**

1. **非官方文档**：本项目完全由 **AI（人工智能）辅助汉化、批注、代码走读与长篇技术解析**，**绝非 NVIDIA 官方文档，亦非官方授权译本**。
2. **准确性声明**：AI 在长文本翻译与复杂数学代数推演中**不保证 100% 绝对准确**，可能存在语义偏差、术语不当或局部推导缺陷。
3. **权威准则**：在生产环境使用或进行关键性能调优时，请务必以 [NVIDIA CUTLASS 官方 GitHub 仓库](https://github.com/NVIDIA/cutlass) 的原版英文源码与规范为准！
:::

<div class="tip custom-block" style="padding-top: 12px; padding-bottom: 12px">

### 为什么需要本项目？

在现代大语言模型（LLM）推理与训练加速领域，**NVIDIA CUTLASS 3.x 及 CuTe（CUDA Tensor Extensions）** 是构建硬件物理极限性能 GPU 算子的行业事实标准。

然而，CuTe 文档充斥着高度抽象的数学概念（如单射/双射坐标变换、布局代数微积分）、高阶 C++ 模板元编程与前沿 GPU 微架构细节，官方纯英文文档晦涩难懂，而普通机器翻译则由于缺乏代数语境往往词不达意、完全失真。

本项目旨在为华语社区的 **GPU 系统工程师、AI 编译器开发者、高校体系结构学者与 HPC 算法专家**，提供一套**条理清晰、源码逐行剖析、万字长篇展开、附带内存地址算例**的深度自学宝典。

</div>

---

## 推荐系统化进阶学习路径

<div class="roadmap-container">
  <div class="roadmap-grid">
    <div class="roadmap-card">
      <div class="roadmap-card-header">
        <div class="roadmap-icon-box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="18" height="18" x="3" y="3" rx="2"/>
            <path d="M3 9h18"/>
            <path d="M9 21V9"/>
          </svg>
        </div>
        <h4 class="roadmap-card-title">第一阶段：夯实纯代数基石</h4>
      </div>
      <ul class="roadmap-step-list">
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/00_quickstart">00. 快速入门与心智模型</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/01_layout">01. Layout 核心映射与公理</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/02_layout_algebra">02. 布局代数微积分</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/03_tensor">03. Tensor 内存模型与切片</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/05_swizzle">05. 共享内存 Swizzle 混洗</a></li>
      </ul>
    </div>

    <div class="roadmap-card">
      <div class="roadmap-card-header">
        <div class="roadmap-icon-box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>
          </svg>
        </div>
        <h4 class="roadmap-card-title">第二阶段：硬件张量与切片</h4>
      </div>
      <ul class="roadmap-step-list">
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/0t_mma_atom">08. MMA Atom 硬件矩阵乘</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/0t_copy_atom">09. Copy Atom 硬件搬运</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/04_algorithms">04. 泛型算子算法体系</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/0x_gemm_tutorial">10. 端到端从零手写 GEMM</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/cute/0z_tma_tensors">12. TMA 异步硬件张量</a></li>
      </ul>
    </div>

    <div class="roadmap-card">
      <div class="roadmap-card-header">
        <div class="roadmap-icon-box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
            <path d="M3 3v5h5"/>
            <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
            <path d="M16 21h5v-5"/>
          </svg>
        </div>
        <h4 class="roadmap-card-title">第三阶段：宏观架构与流水线</h4>
      </div>
      <ul class="roadmap-step-list">
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/architecture/cutlass_3x_design">14. 3.x 设计哲学与词汇类型</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/architecture/gemm_api_3x">15. GEMM API 五层金字塔</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/architecture/pipeline">17. 异步流水线与 mbarrier</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/architecture/tile_scheduler">18. 持久化内核与工作瓦片调度</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/architecture/epilogue_evt">19. 尾声访问者树 (EVT) 融合</a></li>
      </ul>
    </div>

    <div class="roadmap-card">
      <div class="roadmap-card-header">
        <div class="roadmap-icon-box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="16" height="16" x="4" y="4" rx="2"/>
            <rect width="6" height="6" x="9" y="9" rx="1"/>
            <path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/>
          </svg>
        </div>
        <h4 class="roadmap-card-title">第四阶段：前沿微架构与生产算子</h4>
      </div>
      <ul class="roadmap-step-list">
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/architecture/efficient_gemm">20. Roofline 调优与访存极限</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/architecture/blackwell_functionality">22. Blackwell SM100 架构前沿</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/architecture/microscaling_fp4">23. 微缩放浮点 MXFP4 / NVFP4</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/workloads/flash_attention">26. FlashAttention-2/3 算子剖析</a></li>
        <li class="roadmap-step-item"><span class="roadmap-step-bullet"></span><a href="/workloads/moe_gemm">27. MoE 混合专家模型分组 GEMM</a></li>
      </ul>
    </div>
  </div>
</div>

---

## 全景文档专著索引

### 第一部分：CuTe 基础与数学代数 (CuTe Foundations & Layout Algebra)

* [**00. CuTe 快速上手与心智模型构建 (Getting Started)**](/cute/00_quickstart)：环境搭建、源码拓扑、单文件可运行 Demo、调试与 TikZ LaTeX 矢量图生成。
* [**01. CuTe Layout 核心抽象与映射公理 (Layout)**](/cute/01_layout)：函数映射本质、静态整数 `Int<N>`、EBO 内存优化、反向字典序与物理地址对照。
* [**02. CuTe 布局代数运算法则 (Layout Algebra)**](/cute/02_layout_algebra)：合并化简（Coalesce）、函数复合（Composition）、正交补（Complement）、逻辑划分（Logical/Zipped/Tiled/Flat Divide）。
* [**03. CuTe Tensor 核心抽象与切片机制 (Tensors)**](/cute/03_tensor)：`Tensor = Engine + Layout`，强类型指针打标（GMEM/SMEM）、通配符 `_` 切片魔术与 TV（Thread-Value）布局分区。
* [**04. CuTe 泛型张量算法体系 (Algorithms)**](/cute/04_algorithms)：数据搬运 `copy` 三级分发、边界保护 `copy_if`、五大模式 `gemm`、尾声操作 `axpby` 与硬件清零 `clear`。
* [**05. 共享内存 Swizzle 混洗代数与 Bank Conflict 消除 (Swizzle)**](/cute/05_swizzle)：XOR 混洗代数 `Swizzle<B, M, S>`、硬件 Bank 访问机理与 128 位无冲突访存模式。
* [**06. 布局压缩与对偶运算 (Compact, Coalesce & Upcast)**](/cute/06_compact_coalesce)：弱紧凑与强紧凑判断、数据宽度升级（Upcast/Downcast）与多维坐标步长归一化。
* [**07. 张量诊断、运行时打印与 TikZ LaTeX 矢量图生成 (Print & Inspect)**](/cute/07_print_inspect)：`print_layout`、`print_tensor` 深度诊断、PTX 寄存器走读与自动化导出学术级 TikZ 矢量渲染。

### 第二部分：硬件原子与数据搬运抽象 (Hardware Atoms & Data Movement)

* [**08. MMA Atom 硬件矩阵乘抽象 (MMA Atom & TiledMMA)**](/cute/0t_mma_atom)：四层抽象模型、Volta QuadPair 逆向工程、Hopper Warpgroup (WGMMA) 核心矩阵与模式置换代数。
* [**09. Copy Atom 硬件搬运抽象 (UniversalCopy, cp.async & ldmatrix)**](/cute/0t_copy_atom)：异步拷贝硬件原语、SMEM 矢量对齐与 `make_tiled_copy` 自动分块编排。
* [**10. CuTe 密集矩阵乘法 (GEMM) 完整实战教程**](/cute/0x_gemm_tutorial)：从基础 `sgemm_1.cu` 到 128 位向量化访存、Padding 消除 Bank Conflict、多阶段流水线与 GETT 泛化。
* [**11. 谓词判定与非对齐边界裁剪 (Predication)**](/cute/0y_predication)：动态形状下的超矩形盒判定，基于恒等坐标张量的影子平铺（Shadow Tiling）与 Stride-0 零开销广播。
* [**12. CuTe TMA 张量与整数模代数 (TMA Tensors)**](/cute/0z_tma_tensors)：解密神秘签名，$\mathbb{Z}$-模步长公理、规范基底 $E$ 与 `@` 语法，构建 Hopper 硬件 DMA 数据流。
* [**13. 线程块集群协同平铺 (Cluster Tiling & DSMEM)**](/cute/0w_cluster_tiling)：Hopper/Blackwell Threadblock Cluster、分布式共享内存（DSMEM）多播与多块协同计算。

### 第三部分：CUTLASS 3.x 核心架构与流水线 (Architecture & Pipelines)

* [**14. CUTLASS 3.x 核心架构演进与设计哲学 (Design)**](/architecture/cutlass_3x_design)：从 2.x 僵化绑定到 3.x 算法解耦、单一词汇类型设计、标签分发（Tag-Dispatch）与“构造即正确”保证。
* [**15. CUTLASS 3.x GEMM API 体系与五层架构全景 (GEMM API)**](/architecture/gemm_api_3x)：Device 到 Kernel 到 Collective 到 Tiled 到 Atom 五层金字塔、`CollectiveBuilder` 自动合成与 EVT 算子融合树。
* [**16. CollectiveBuilder 自动元编程编译器深度解析 (CollectiveBuilder)**](/architecture/collective_builder)：类型生成管道、模板特化匹配与根据硬件平台自动推荐最优内核配置。
* [**17. CUTLASS 异步软件流水线与同步机制 (Pipeline)**](/architecture/pipeline)：利特尔法则延迟隐藏、Warp Specialization 生产者-消费者解耦、`mbarrier` 硬件事务计数与相位反转无锁同步。
* [**18. 持久化内核与工作瓦片调度器 (Tile Scheduler)**](/architecture/tile_scheduler)：Persistent Grid 模型、Stream-K 动态负载均衡、Raster-Order 缓存局部性优化。
* [**19. 尾声访问者树原地算子融合 (Epilogue Visitor Tree)**](/architecture/epilogue_evt)：无开销中间变量写回、计算图树状编译、Bias+GELU/SiLU 原地融合流水线。
* [**20. 高性能 GEMM 硬件物理极限与优化法则 (Efficient GEMM)**](/architecture/efficient_gemm)：Roofline 模型、三级嵌套分块、L2 缓存蛇形混洗、Split-K / Sliced-K 极端长条矩阵优化与 NCU 调优清单。
* [**21. 极端长条矩阵并行归约 (Split-K & Sliced-K)**](/architecture/split_k)：批量长条矩阵计算瓶颈、跨线程块原子归约、两阶段 Tree-Reduction 与精度损失控制。

### 第四部分：硬件微架构前沿 (GPU Microarchitectures & Blackwell)

* [**22. NVIDIA Blackwell (SM100) 架构核心特性 (Blackwell Functionality)**](/architecture/blackwell_functionality)：第 5 代 Tensor Core `tcgen05.mma`、OCP MXFP4 / NVFP4 块级微缩放（Microscaling）与双 SM 协同执行（2SM Cooperative）。
* [**23. 微缩放浮点格式原理与实现 (Microscaling FP4/FP8)**](/architecture/microscaling_fp4)：E2M1 / E4M3 微缩放规范、Block Scale Factor 计算、动态动态范围补偿与硬件解码通道。
* [**24. 2-SM 硬件集群协同执行机制 (2-SM Cooperative & cta_group::2)**](/architecture/two_sm_cooperative)：跨 SM 寄存器交换与 Tensor Core 资源倍增、硬件级集群网格调度与延迟重叠。
* [**25. 2:4 结构化稀疏硬件加速 (Structured Sparsity GEMM)**](/architecture/sparse_gemm)：Ampere 到 Blackwell 稀疏 Tensor Core 特性、元数据压缩编码格式与稀疏矩阵高效构建。

### 第五部分：大模型工业级前沿算子实战 (LLM & Production Workloads)

* [**26. FlashAttention-2/3 底层 CuTe 实现原理解析 (FlashAttention)**](/workloads/flash_attention)：Online Softmax 数值稳定性推导、Q/K/V 分块内存布局、WGMMA 异步计算与 TMA 流水线编排。
* [**27. 混合专家模型分组 GEMM 架构 (MoE Grouped GEMM)**](/workloads/moe_gemm)：动态路由 Token 负载不均问题、Grouped GEMM 连续与离散内存调度、CUDA Graph 兼容模式。
* [**28. 隐式 GEMM 卷积算子实现 (Implicit GEMM Conv2d)**](/workloads/conv2d_as_gemm)：无额外 im2col 内存开销、高维张量坐标变换与分步卷积前向/反向高效计算。
* [**29. W4A16 / W8A8 混合精度大模型量化算子 (Quantized GEMM)**](/workloads/quantized_gemm)：AWQ / GPTQ 工业级算子实现、寄存器内实时解包反量化（De-quantization）与双数据流重叠。
* [**30. LayerNorm / RMSNorm + GEMM 端到端算子深度融合 (Fused Norm GEMM)**](/workloads/layernorm_gemm_fusion)：跨算子内存往返消除、Warp Shuffle 均值方差归约、共享内存驻留与单内核完成整网层前向传播。

### 第六部分：权威参考与词典

* [**核心术语与代数符号权威词典 (Glossary)**](/glossary)：统一的中英双语术语对照与形式化数学符号检索手册。
