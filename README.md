# NVIDIA CUTLASS 3.x & CuTe 核心架构学习指南（AI 辅助汉化与深度解析版）

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%20%7C%2013.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Architecture](https://img.shields.io/badge/Architecture-Volta%20%7C%20Ampere%20%7C%20Hopper%20%7C%20Blackwell-purple.svg)](https://github.com/NVIDIA/cutlass)
[![VitePress](https://img.shields.io/badge/Powered%20by-VitePress-orange.svg)](https://vitepress.dev/)

**系统化、工程级、万字长文深入剖析 NVIDIA CUTLASS 3.x 与 CuTe 底层算子与架构规范**

[📖 在线文档目录](#-全套文档全景导读) • [📐 核心术语表](./docs/glossary.md) • [🚀 本地运行与构建](#-本地运行与构建指南)

</div>

---

> [!WARNING] ⚠️ 免责与非官方声明
> * **非官方出品**：本项目完全由 **AI（人工智能）辅助汉化、批注、代码走读与原理展开**，**非 NVIDIA 官方文档，亦非官方授权译本**。
> * **内容不保证完全准确**：在处理极为复杂的 GPU 微架构指令（如 TMA、Warpgroup MMA、微缩放格式）与代数推导时，翻译及解读可能存在疏漏或瑕疵，**切勿将其作为绝对标准**。
> * **权威溯源**：所有技术实现与权威依据，请一律以 [NVIDIA CUTLASS 官方 GitHub 仓库](https://github.com/NVIDIA/cutlass) 的原版英文源码和文档为准。

---

## 🌟 为什么发起本项目？

在当今生成式 AI、大语言模型（LLM）推理与训练算子加速领域（如 **FlashAttention-2/3**、**FP8/FP4 量化**、**DeepSeek 极致优化内核**），**NVIDIA CUTLASS 3.x 及 CuTe（CUDA Tensor Extensions）** 是构建硬件极致性能 GPU 算子的行业事实标准。

然而，CuTe 文档由于充斥着**高度抽象的数学代数（如高维单射映射、布局代数微积分）、现代 C++ 模板元编程与前沿 GPU 微架构底层细节**，官方全英文文档极其晦涩难懂，而普通机器翻译（如 Google/DeepL）在缺乏代数语境下往往词不达意、完全失真。

国内众多 GPU 算子工程师、AI 编译器研究员以及高校体系结构学者，长期受困于零碎散落的技术博客，迫切需要一套**结构化展开、数学推导透彻、源码逐行剖析、万字长文解析**的学习参考资料。

本项目旨在系统性地降低理解门槛，助力华语开发者攻克 GPU 底层核心算子技术。

---

## 📚 全套文档全景导读

### 一、CuTe 核心规范与布局代数 (CuTe Core Specifications)
* [**00. 快速入门指南 (Quickstart)**](./docs/cute/00_quickstart.md)：CuTe 系统架构、代码组织、设备端高效调试与打印机制。
* [**01. Layout 核心抽象 (Layout)**](./docs/cute/01_layout.md)：揭示 Layout 的数学本质（整数到整数函数）、Shape 与 Stride、动态/静态整数统一、反向字典序（Colexicographical Order）。
* [**02. 布局代数运算法则 (Layout Algebra)**](./docs/cute/02_layout_algebra.md)：化简（Coalesce）、函数复合（Composition）、补全（Complement）、逻辑划分（Logical/Zipped/Tiled Divide）与布局乘法。
* [**03. Tensor 抽象与切片 (Tensor)**](./docs/cute/03_tensor.md)：`Tensor = Engine + Layout`，显存空间打标（GMEM/SMEM/Register）、切片通配符 `_`、内部分区、外部分区与 TV 布局。
* [**04. 泛型张量算法体系 (Algorithms)**](./docs/cute/04_algorithms.md)：`copy`、`copy_if`、`gemm`（5 大计算范式）、`axpby`、`fill` 与 `clear`。
* [**05. MMA Atom 硬件矩阵乘抽象 (MMA Atom)**](./docs/cute/0t_mma_atom.md)：从 Volta QuadPair 到 Hopper Warpgroup (WGMMA) 核心矩阵推导，TiledMMA 线程与数值平铺。
* [**06. 手把手实现极速 GEMM 教程 (GEMM Tutorial)**](./docs/cute/0x_gemm_tutorial.md)：从单线程直译到 128 位向量化访存、共享内存 Padding 消除 Bank Conflict、GETT 高维张量缩并。
* [**07. 谓词判定与边缘裁剪 (Predication)**](./docs/cute/0y_predication.md)：解决工业场景非对齐矩阵乘，基于恒等坐标张量的影子平铺（Shadow Tiling）与零开销掩码。
* [**08. TMA 异步硬件张量拷贝 (TMA Tensors)**](./docs/cute/0z_tma_tensors.md)：针对 Hopper/Blackwell 硬件 TMA，揭秘整数模步长代数（$\mathbb{Z}$-Module）与规范基底 `E`。

### 二、CUTLASS 3.x 架构与前沿特性 (Architecture & Modern GPUs)
* [**1. CUTLASS 3.x 核心设计理念 (Design)**](./docs/architecture/cutlass_3x_design.md)：从 2.x 硬件僵化绑定到 3.x 算法解耦、消灭冗余具名类、标签分发策略（Tag-dispatch Policies）。
* [**2. GEMM API 3.x 详解 (GEMM API)**](./docs/architecture/gemm_api_3x.md)：Device $\to$ Kernel $\to$ Collective $\to$ Tiled $\to$ Atom 五层金字塔架构与算子组装四部曲。
* [**3. 异步软件流水线机制 (Pipeline)**](./docs/architecture/pipeline.md)：延迟隐藏技术、生产者-消费者四步闭环协议、Thread Block Cluster 硬件栅障。
* [**4. 高性能 GEMM 硬件理论与优化 (Efficient GEMM)**](./docs/architecture/efficient_gemm.md)：三层层次化分块、Split-K/Sliced-K 极端矩阵优化、Hopper Warp-Specialization（Cooperative & Ping-Pong 乒乓架构）。
* [**5. Blackwell (SM100) 架构核心支持 (Blackwell Functionality)**](./docs/architecture/blackwell_functionality.md)：第 5 代 Tensor Core `tcgen05.mma`、OCP MX / NVFP4 块级微缩放（Block-scaled）、双 SM 硬件协同。

### 三、工程参考基石
* [**📐 核心术语与代数符号对照表 (Glossary)**](./docs/glossary.md)：全套文档的统一术语基准。

---

## 🚀 本地运行与构建指南

本项目基于业界主流的现代静态文档生成器 [VitePress](https://vitepress.dev/) 构建，原生支持 Markdown、KaTeX 数学公式渲染、代码一键复制与深色模式。

### 1. 安装依赖并启动本地预览
```bash
pnpm install
pnpm run docs:dev
```
启动后在浏览器打开 `http://localhost:5173` 即可享受极致的本地阅读体验。

### 2. 构建静态产物
```bash
pnpm run docs:build
```
编译生成的静态 HTML 文件位于 `docs/.vitepress/dist` 目录下。

---

## 🤝 致谢与开源协议

* 本项目遵循 **[Apache-2.0 License](LICENSE)** 开源许可证。
* 文档原文与核心技术知识产权完全归属于 **NVIDIA Corporation** 及其开源贡献者。
* 欢迎广大 GPU 系统开发者提交 Issue、校对勘误或提交 Pull Request。
