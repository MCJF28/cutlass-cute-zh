import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
  title: "CUTLASS 3.x & CuTe 学习指南",
  description: "NVIDIA CUTLASS 3.x 与 CuTe 核心架构学习指南（AI 辅助汉化与深度解析版）",
  base: process.env.BASE_PATH || (process.env.GITHUB_ACTIONS ? '/cutlass-cute-zh/' : '/'),
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }]
  ],
  themeConfig: {
    nav: [
      { text: '首页', link: '/', activeMatch: '^/$' },
      { text: 'CuTe 代数规范', link: '/cute/00_quickstart', activeMatch: '^/cute/0[0-7]' },
      { text: '硬件原子与搬运', link: '/cute/0t_mma_atom', activeMatch: '^/cute/(0t|0x|0y|0z|0w)' },
      { text: '3.x 核心架构', link: '/architecture/cutlass_3x_design', activeMatch: '^/architecture/(cutlass_3x_design|gemm_api_3x|collective_builder|pipeline|tile_scheduler|epilogue_evt|efficient_gemm|split_k)' },
      { text: '微架构与 Blackwell', link: '/architecture/blackwell_functionality', activeMatch: '^/architecture/(blackwell_functionality|microscaling_fp4|two_sm_cooperative|sparse_gemm)' },
      { text: '大模型前沿算子', link: '/workloads/flash_attention', activeMatch: '^/workloads/' },
      { text: '术语词典', link: '/glossary', activeMatch: '^/glossary' },
      { text: 'NVIDIA 源码', link: 'https://github.com/NVIDIA/cutlass' }
    ],
    sidebar: {
      // Common comprehensive sidebar across all document sections
      '/': [
        {
          text: '第一部分：CuTe 基础与数学代数',
          collapsed: false,
          items: [
            { text: '00. 快速入门与心智模型', link: '/cute/00_quickstart' },
            { text: '01. Layout 核心映射与公理', link: '/cute/01_layout' },
            { text: '02. 布局代数微积分', link: '/cute/02_layout_algebra' },
            { text: '03. Tensor 内存模型与切片', link: '/cute/03_tensor' },
            { text: '04. 泛型数据流算法体系', link: '/cute/04_algorithms' },
            { text: '05. 共享内存 Swizzle 混洗代数', link: '/cute/05_swizzle' },
            { text: '06. 布局压缩与对偶运算', link: '/cute/06_compact_coalesce' },
            { text: '07. 张量诊断与 TikZ 矢量渲染', link: '/cute/07_print_inspect' },
          ]
        },
        {
          text: '第二部分：硬件原子与数据搬运',
          collapsed: false,
          items: [
            { text: '08. MMA Atom 硬件矩阵乘抽象', link: '/cute/0t_mma_atom' },
            { text: '09. Copy Atom 硬件搬运抽象', link: '/cute/0t_copy_atom' },
            { text: '10. 手把手从零手写 GEMM 实战', link: '/cute/0x_gemm_tutorial' },
            { text: '11. 谓词判定与非对齐边界裁剪', link: '/cute/0y_predication' },
            { text: '12. TMA 异步硬件张量与整数模代数', link: '/cute/0z_tma_tensors' },
            { text: '13. 线程块集群协同平铺 (Cluster)', link: '/cute/0w_cluster_tiling' },
          ]
        },
        {
          text: '第三部分：CUTLASS 3.x 核心架构与流水线',
          collapsed: false,
          items: [
            { text: '14. 3.x 设计哲学与单一词汇类型', link: '/architecture/cutlass_3x_design' },
            { text: '15. GEMM API 3.x 五层架构金字塔', link: '/architecture/gemm_api_3x' },
            { text: '16. CollectiveBuilder 自动元编程编译器', link: '/architecture/collective_builder' },
            { text: '17. 异步软件流水线与多阶段同步', link: '/architecture/pipeline' },
            { text: '18. 持久化内核与工作瓦片调度器', link: '/architecture/tile_scheduler' },
            { text: '19. 尾声访问者树 (EVT) 原地融合', link: '/architecture/epilogue_evt' },
            { text: '20. 高性能 GEMM 理论与 Roofline 极限', link: '/architecture/efficient_gemm' },
            { text: '21. 极端长条矩阵并行归约 (Split-K)', link: '/architecture/split_k' },
          ]
        },
        {
          text: '第四部分：硬件微架构前沿 (Blackwell & SM100)',
          collapsed: false,
          items: [
            { text: '22. NVIDIA Blackwell 硬件特性详解', link: '/architecture/blackwell_functionality' },
            { text: '23. 微缩放浮点格式原理 (MXFP4/NVFP4)', link: '/architecture/microscaling_fp4' },
            { text: '24. 2-SM 硬件集群协同执行机制', link: '/architecture/two_sm_cooperative' },
            { text: '25. 2:4 结构化稀疏硬件加速', link: '/architecture/sparse_gemm' },
          ]
        },
        {
          text: '第五部分：大模型工业级前沿算子实战',
          collapsed: false,
          items: [
            { text: '26. FlashAttention-2/3 底层 CuTe 解析', link: '/workloads/flash_attention' },
            { text: '27. 混合专家模型 (MoE) 分组 GEMM', link: '/workloads/moe_gemm' },
            { text: '28. 隐式 GEMM 卷积算子实现', link: '/workloads/conv2d_as_gemm' },
            { text: '29. W4A16 / W8A8 混合精度量化算子', link: '/workloads/quantized_gemm' },
            { text: '30. LayerNorm / RMSNorm + GEMM 深度融合', link: '/workloads/layernorm_gemm_fusion' },
          ]
        },
        {
          text: '第六部分：权威参考与词典',
          collapsed: false,
          items: [
            { text: '31. 核心术语与代数符号权威词典', link: '/glossary' }
          ]
        }
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/NVIDIA/cutlass' }
    ],
    footer: {
      message: '免责声明：本网站只是用来消耗反重力额度的，并且给有需要的人看一看，准确度极低。可用度极低，不想看的请不看，造成错误概不负责。',
      copyright: 'Copyright © 2026 NVIDIA Corporation (Original) & AI Community Contributors'
    },
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档、算法与架构...',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '未找到相关技术文档',
                resetButtonTitle: '清除搜索',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭'
                }
              }
            }
          }
        }
      }
    },
    outline: {
      level: [2, 3],
      label: '本页目录'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    }
  },
  vite: {
    optimizeDeps: {
      include: ['dayjs', 'mermaid']
    }
  },
  mermaid: {
    theme: 'default'
  }
}))
