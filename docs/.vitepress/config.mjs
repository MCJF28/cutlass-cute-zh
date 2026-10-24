import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "CUTLASS 3.x & CuTe 学习指南",
  description: "NVIDIA CUTLASS 3.x 与 CuTe 核心架构学习指南（AI 辅助汉化与深度解析版）",
  base: "/",
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }]
  ],
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '术语表 (Glossary)', link: '/glossary' },
      { text: 'CuTe 核心规范', link: '/cute/00_quickstart' },
      { text: 'CUTLASS 3.x 架构', link: '/architecture/cutlass_3x_design' },
      { text: 'NVIDIA 官方原版代码', link: 'https://github.com/NVIDIA/cutlass' }
    ],
    sidebar: {
      '/cute/': [
        {
          text: 'CuTe 核心规范与深度解析',
          items: [
            { text: '00. 快速入门与环境 (Quickstart)', link: '/cute/00_quickstart' },
            { text: '01. Layout 核心抽象与映射 (Layout)', link: '/cute/01_layout' },
            { text: '02. 布局代数微积分 (Layout Algebra)', link: '/cute/02_layout_algebra' },
            { text: '03. Tensor 内存模型与切片 (Tensor)', link: '/cute/03_tensor' },
            { text: '04. 泛型算子算法体系 (Algorithms)', link: '/cute/04_algorithms' },
            { text: '05. MMA Atom 硬件矩阵乘抽象', link: '/cute/0t_mma_atom' },
            { text: '06. 手把手极速 GEMM 实战全流程', link: '/cute/0x_gemm_tutorial' },
            { text: '07. 谓词判定与边缘裁剪 (Predication)', link: '/cute/0y_predication' },
            { text: '08. TMA 异步硬件张量拷贝', link: '/cute/0z_tma_tensors' },
          ]
        }
      ],
      '/architecture/': [
        {
          text: 'CUTLASS 3.x 架构深度解析',
          items: [
            { text: '3.x 总体设计理念 (Design)', link: '/architecture/cutlass_3x_design' },
            { text: 'GEMM API 3.x 五层架构详解', link: '/architecture/gemm_api_3x' },
            { text: '异步软件流水线 (Pipeline)', link: '/architecture/pipeline' },
            { text: '高性能 GEMM 理论与优化', link: '/architecture/efficient_gemm' },
            { text: 'Blackwell 架构硬件特性支持', link: '/architecture/blackwell_functionality' },
          ]
        }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/NVIDIA/cutlass' }
    ],
    footer: {
      message: '免责声明：本项目完全由 AI 辅助汉化与解读，非 NVIDIA 官方译本，不保证 100% 准确无误。请以 NVIDIA 官方英文原文为准。',
      copyright: 'Copyright © 2026 NVIDIA Corporation (Original) & AI Community Contributors'
    },
    search: {
      provider: 'local'
    },
    outline: {
      level: [2, 3],
      label: '本页目录'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    }
  }
})
