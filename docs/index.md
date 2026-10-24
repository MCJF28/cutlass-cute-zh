---
layout: home

hero:
  name: "CUTLASS 3.x & CuTe"
  text: "底层高性能 GPU 算子开发与核心架构解析"
  tagline: "全网系统化超详尽解析指南（AI 辅助汉化与深度剖析版）"
  actions:
    - theme: brand
      text: 快速开始 (Quickstart)
      link: /cute/00_quickstart
    - theme: alt
      text: 核心术语对照表 (Glossary)
      link: /glossary
    - theme: alt
      text: 手把手极速 GEMM 教程
      link: /cute/0x_gemm_tutorial

features:
  - icon: 📐
    title: 布局代数 (Layout Algebra)
    details: 彻底拆解 Shape、Stride、分层嵌套布局、Composition、Coalesce 与 Complement，将高维张量坐标精准映射至物理内存。
  - icon: ⚡
    title: 硬件级 MMA 与 TMA 抽象
    details: 深入 Hopper 与 Blackwell 架构的 WGMMA (Warpgroup MMA)、Tensor Memory Accelerator (TMA) 异步硬件加速通道。
  - icon: 🛠️
    title: 现代算子核心基石
    details: FlashAttention-2/3、大模型 FP8/FP4 混合精度量化、DeepSeek 极致算子优化的底层支撑核心。
  - icon: 📖
    title: 超详尽剖析与代码走读
    details: 配备万字长文级别的深入推导、内存布局 ASCII 可视化、PTX 汇编映射与避坑指南。
---

::: warning ⚠️ 重要免责与非官方声明
1. **非官方文档**：本项目完全由 **AI（人工智能）辅助汉化、批注与长文深度剖析**，**绝非 NVIDIA 官方文档，亦非官方授权译本**。
2. **准确性声明**：AI 在长文本翻译与复杂数学代数推演中**不保证 100% 绝对准确**，可能存在语义偏差、术语不当或局部推导缺陷。
3. **权威准则**：在生产环境使用或进行关键性能调优时，请务必以 [NVIDIA CUTLASS 官方 GitHub 仓库](https://github.com/NVIDIA/cutlass) 的原版英文源码与规范为准！
:::

<div class="tip custom-block" style="padding-top: 8px">

### 💡 为什么需要本项目？

在现代大语言模型（LLM）推理与训练加速领域，**NVIDIA CUTLASS 3.x 及 CuTe（CUDA Tensor Extensions）** 是构建极致性能 GPU 算子的行业标准。

然而，CuTe 文档充斥着高度抽象的数学概念（如单射/双射坐标变换、布局代数微积分）、高阶 C++ 模板元编程与前沿 GPU 微架构细节，官方纯英文文档晦涩难懂，而普通机器翻译则由于缺乏代数语境往往词不达意。

本项目旨在为华语社区的 **GPU 系统工程师、AI 编译器开发者、高校体系结构学者与 HPC 算法专家**，提供一套**条理清晰、源码逐行剖析、万字长文展开、附带内存地址算例**的深度自学宝典。
</div>
