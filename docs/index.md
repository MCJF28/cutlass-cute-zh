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

<RoadmapGrid />

---

## 全景文档专著索引

<DocIndexCatalog />
