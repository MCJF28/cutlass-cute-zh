<template>
  <div class="doc-catalog-wrapper">
    <div v-for="(section, sIdx) in catalogData" :key="sIdx" class="catalog-section">
      <!-- Section Header -->
      <div class="catalog-section-header">
        <div class="section-title-group">
          <div class="section-number-badge">Part {{ sIdx + 1 }}</div>
          <h3 class="section-title">{{ section.title }}</h3>
        </div>
        <div class="section-meta">
          <span class="section-desc">{{ section.description }}</span>
          <span class="section-count-pill">{{ section.articles.length }} 篇专著</span>
        </div>
      </div>

      <!-- Articles Grid -->
      <div class="catalog-grid">
        <a 
          v-for="(art, aIdx) in section.articles" 
          :key="aIdx" 
          :href="art.link" 
          class="catalog-card"
        >
          <div class="card-header">
            <span class="chapter-badge">{{ art.chapter }}</span>
            <span v-if="art.tag" class="topic-tag" :class="art.tagType">{{ art.tag }}</span>
          </div>

          <h4 class="card-title">
            <span>{{ art.title }}</span>
            <svg class="card-arrow-icon" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </h4>

          <p class="card-details">{{ art.details }}</p>

          <div class="card-footer" v-if="art.keywords && art.keywords.length">
            <span v-for="(kw, kIdx) in art.keywords" :key="kIdx" class="card-kw-pill">
              {{ kw }}
            </span>
          </div>
        </a>
      </div>
    </div>
  </div>
</template>

<script setup>
const catalogData = [
  {
    title: '第一部分：CuTe 基础与数学代数 (Foundations & Layout Algebra)',
    description: '深入纯代数公理、Layout 映射映射机制、坐标微积分与无冲突混洗',
    articles: [
      {
        chapter: '00',
        title: 'CuTe 快速上手与心智模型构建',
        link: '/cute/00_quickstart',
        tag: '快速入门',
        tagType: 'tag-blue',
        details: '彻底打破传统 CUDA 扁平指针起算心智，建立 Layout 映射坐标系，包含单文件 Demo 与环境搭建。',
        keywords: ['环境拓扑', '心智重构', 'print_layout', 'TikZ 矢量图']
      },
      {
        chapter: '01',
        title: 'Layout 核心抽象与映射公理',
        link: '/cute/01_layout',
        tag: '核心代数',
        tagType: 'tag-purple',
        details: '函数映射本质、静态常量 Int<N>、EBO 内存空基类优化、反向字典序与物理地址一一对照。',
        keywords: ['Shape', 'Stride', 'Colexicographical', '物理内存对照']
      },
      {
        chapter: '02',
        title: '布局代数运算法则 (微积分)',
        link: '/cute/02_layout_algebra',
        tag: '代数进阶',
        tagType: 'tag-purple',
        details: '严密推演合并化简 (Coalesce)、函数复合 (Composition)、正交补 (Complement) 与逻辑划分。',
        keywords: ['Coalesce', 'Composition', 'Complement', 'Logical Divide']
      },
      {
        chapter: '03',
        title: 'Tensor 核心抽象与切片机制',
        link: '/cute/03_tensor',
        tag: '张量模型',
        tagType: 'tag-blue',
        details: 'Tensor = Engine + Layout，GMEM/SMEM 强类型指针打标，通配符切片与 TV (Thread-Value) 布局。',
        keywords: ['Engine', 'Pointer-Tagging', '通配符 _', 'TV 分区']
      },
      {
        chapter: '04',
        title: '泛型张量算法体系 (Algorithms)',
        link: '/cute/04_algorithms',
        tag: '泛型算法',
        tagType: 'tag-green',
        details: '数据搬运 copy 三级分发、边界保护 copy_if、五大模式 gemm、axpby 与硬件清零 clear。',
        keywords: ['copy 三级分发', 'copy_if', 'gemm 5 大模式', 'clear']
      },
      {
        chapter: '05',
        title: '共享内存 Swizzle 混洗代数',
        link: '/cute/05_swizzle',
        tag: '冲突消除',
        tagType: 'tag-amber',
        details: '32-Bank 硬件机理、XOR 混洗代数 Swizzle<B, M, S>、128 位向量化访问与零冲突排布。',
        keywords: ['32-Bank', 'Swizzle<B,M,S>', '零内存浪费', '128B 对齐']
      },
      {
        chapter: '06',
        title: '布局压缩与对偶运算',
        link: '/cute/06_compact_coalesce',
        tag: '优化工具',
        tagType: 'tag-purple',
        details: '严格紧凑与弱紧凑代数公理、coalesce 连续维度折叠、upcast/downcast 数据类型位宽缩放。',
        keywords: ['紧凑性公理', 'upcast<N>', 'downcast<N>', '步长归一']
      },
      {
        chapter: '07',
        title: '张量诊断与 TikZ LaTeX 矢量生成',
        link: '/cute/07_print_inspect',
        tag: '内省调试',
        tagType: 'tag-green',
        details: '编译期类型推导诊断、终端二维 ASCII 网格直观打印与自动导出学术级 TikZ 矢量排布图。',
        keywords: ['print_layout', 'thread0()', 'TikZ 导出', '类型签名']
      }
    ]
  },
  {
    title: '第二部分：硬件原子与数据搬运抽象 (Hardware Atoms & Data Movement)',
    description: '掌握 Tensor Core 硬件计算原子、异步 DMA 传输通道与跨块协同集群',
    articles: [
      {
        chapter: '08',
        title: 'MMA Atom 硬件矩阵乘抽象',
        link: '/cute/0t_mma_atom',
        tag: '计算原子',
        tagType: 'tag-purple',
        details: '四层抽象模型、Volta QuadPair 逆向工程、Hopper Warpgroup (WGMMA) 核心与模式置换。',
        keywords: ['MMA_Atom', 'TiledMMA', 'WGMMA', 'QuadPair']
      },
      {
        chapter: '09',
        title: 'Copy Atom 硬件搬运抽象',
        link: '/cute/0t_copy_atom',
        tag: '搬运原语',
        tagType: 'tag-blue',
        details: 'UniversalCopy、Ampere cp.async 异步拷贝、ldmatrix 转置指令与 make_tiled_copy 宏平铺。',
        keywords: ['cp.async', 'ldmatrix', 'partition_S', 'make_tiled_copy']
      },
      {
        chapter: '10',
        title: 'CuTe 密集矩阵乘 (GEMM) 实战教程',
        link: '/cute/0x_gemm_tutorial',
        tag: '实战进阶',
        tagType: 'tag-green',
        details: '从 sgemm_1 裸写逐步迭代到 128 位向量访存、Padding 消除 Bank 冲突、多阶段流水线与 GETT。',
        keywords: ['从零手写', '向量化对齐', 'Padding 消除', 'GETT 泛化']
      },
      {
        chapter: '11',
        title: '谓词判定与非对齐边界裁剪',
        link: '/cute/0y_predication',
        tag: '边界保护',
        tagType: 'tag-amber',
        details: '动态任意矩阵尺寸超矩形盒判定，基于恒等坐标张量的影子平铺 (Shadow Tiling) 与零开销广播。',
        keywords: ['Shadow Tiling', 'Identity Tensor', 'copy_if', '硬件谓词']
      },
      {
        chapter: '12',
        title: 'CuTe TMA 张量与整数模代数',
        link: '/cute/0z_tma_tensors',
        tag: 'Hopper 硬件',
        tagType: 'tag-purple',
        details: '解密 TMA 神秘类型签名，Z-模步长公理、规范基底 E 与 @ 语法，构建纯硬件异步数据通道。',
        keywords: ['TMA Descriptor', 'Z-模公理', '规范基底 E', '无寄存器中转']
      },
      {
        chapter: '13',
        title: '线程块集群协同平铺 (Cluster Tiling)',
        link: '/cute/0w_cluster_tiling',
        tag: '集群架构',
        tagType: 'tag-blue',
        details: 'Hopper/Blackwell Threadblock Cluster、分布式共享内存 (DSMEM) 片上互联与 TMA 多播广播。',
        keywords: ['DSMEM', 'TMA Multicast', 'NoC 高速互联', '跨 SM 寻址']
      }
    ]
  },
  {
    title: '第三部分：CUTLASS 3.x 核心架构与流水线 (Architecture & Pipelines)',
    description: '工业级算子库五层金字塔全景、无锁软件流水线、持久化调度与原地算子融合',
    articles: [
      {
        chapter: '14',
        title: '3.x 核心设计理念与单一词汇类型',
        link: '/architecture/cutlass_3x_design',
        tag: '设计哲学',
        tagType: 'tag-blue',
        details: '从 2.x 模板爆炸到 3.x 算法解耦、消灭具名类、标签分发 (Tag-Dispatch) 与构造即正确保证。',
        keywords: ['单一词汇类型', '算法解耦', '标签分发', '编译期校验']
      },
      {
        chapter: '15',
        title: 'GEMM API 3.x 五层架构金字塔全景',
        link: '/architecture/gemm_api_3x',
        tag: '系统架构',
        tagType: 'tag-purple',
        details: 'Device 到 Kernel 到 Collective 到 Tiled 到 Atom 五层金字塔、参数打包解包与构造四部曲。',
        keywords: ['Device Layer', 'Kernel Layer', 'Collective', 'Adapter']
      },
      {
        chapter: '16',
        title: 'CollectiveBuilder 元编程编译器',
        link: '/architecture/collective_builder',
        tag: '元编程编译',
        tagType: 'tag-blue',
        details: '声明式参数输入、硬件微架构自动分发、阶段数智能搜索与全自动合成最优 Mainloop。',
        keywords: ['StageCountAuto', '声明式编程', '调度推导', '主循环合成']
      },
      {
        chapter: '17',
        title: '异步软件流水线与多阶段同步',
        link: '/architecture/pipeline',
        tag: '并发流水',
        tagType: 'tag-amber',
        details: '利特尔法则延迟隐藏、Warp Specialization 生产者-消费者解耦、mbarrier 事务屏障与相位反转。',
        keywords: ['Warp Specialization', 'mbarrier', 'Phase Flip', '无锁同步']
      },
      {
        chapter: '18',
        title: '持久化内核与工作瓦片调度器',
        link: '/architecture/tile_scheduler',
        tag: '调度突破',
        tagType: 'tag-green',
        details: 'Persistent Grid 常驻线程块模型、消灭尾部效应的 Stream-K 均分算法与 L2 缓存蛇形光栅化。',
        keywords: ['Persistent Grid', 'Stream-K', '尾部效应', 'Snake Walk']
      },
      {
        chapter: '19',
        title: '尾声访问者树 (EVT) 原地算子融合',
        link: '/architecture/epilogue_evt',
        tag: '算子融合',
        tagType: 'tag-purple',
        details: '消灭显存往返、编译期 AST 语法树编排、Bias+GELU/SiLU+Residual 寄存器堆原地计算。',
        keywords: ['EVT 抽象语法树', '原地融合', '零显存中转', '多头写回']
      },
      {
        chapter: '20',
        title: '高性能 GEMM 理论与 Roofline 极限',
        link: '/architecture/efficient_gemm',
        tag: '调优宝典',
        tagType: 'tag-amber',
        details: 'Roofline 模型深度推导、三级层次化分块平衡点、NCU 性能瓶颈诊断与调优 Checklist。',
        keywords: ['Roofline', '三级分块', 'NCU 调优', 'FLOPS/Byte']
      },
      {
        chapter: '21',
        title: '极端长条矩阵并行归约 (Split-K)',
        link: '/architecture/split_k',
        tag: '长条矩阵',
        tagType: 'tag-blue',
        details: '破解大模型单 Token 解码 SM 空转危机、Serial Split-K 原子写回与 Parallel Split-K 两阶段归约。',
        keywords: ['Tall-Skinny', 'Serial Split-K', 'Parallel Split-K', '原子加']
      }
    ]
  },
  {
    title: '第四部分：硬件微架构前沿 (GPU Microarchitectures & Blackwell)',
    description: '探索 NVIDIA Blackwell SM100、NVFP4 微缩放、双 SM 硬件协同与稀疏加速',
    articles: [
      {
        chapter: '22',
        title: 'NVIDIA Blackwell (SM100) 硬件特性',
        link: '/architecture/blackwell_functionality',
        tag: 'Blackwell',
        tagType: 'tag-purple',
        details: '第 5 代 Tensor Core tcgen05、NVFP4 块级微缩放、2-SM 协同执行与前沿计算拓扑。',
        keywords: ['SM100', 'tcgen05', 'Microscaling', 'cta_group::2']
      },
      {
        chapter: '23',
        title: '微缩放浮点格式原理 (MXFP4/NVFP4)',
        link: '/architecture/microscaling_fp4',
        tag: '低精度前沿',
        tagType: 'tag-amber',
        details: 'E2M1 编码、32 元素微块局域独立缩放因子、硬件飞线解码与算力暴增 2 倍。',
        keywords: ['E2M1', 'Block Scale', '飞线解压', '20 PFLOPS']
      },
      {
        chapter: '24',
        title: '2-SM 硬件集群协同执行机制',
        link: '/architecture/two_sm_cooperative',
        tag: '硬件突破',
        tagType: 'tag-purple',
        details: '打破单 SM 物理隔离、跨 SM 寄存器堆对角线互联总线、cta_group::2 双联体调度。',
        keywords: ['cta_group::2', '跨 SM 互联', '寄存器容量翻倍', '92% MFU']
      },
      {
        chapter: '25',
        title: '2:4 结构化稀疏硬件加速',
        link: '/architecture/sparse_gemm',
        tag: '稀疏计算',
        tagType: 'tag-green',
        details: '剪枝加速悖论突破、2:4 局部几何约束、2-bit 元数据压缩编码与 Sparse Tensor Core 吞吐翻倍。',
        keywords: ['2:4 稀疏', '元数据压缩', 'Sparse Tensor Core', 'Wanda / SparseGPT']
      }
    ]
  },
  {
    title: '第五部分：大模型工业级前沿算子实战 (LLM Production Workloads)',
    description: '工业界顶级核心算子深度解析：FlashAttention-2/3、MoE、隐式卷积与量化融合',
    articles: [
      {
        chapter: '26',
        title: 'FlashAttention-2/3 底层 CuTe 解析',
        link: '/workloads/flash_attention',
        tag: '大模型核心',
        tagType: 'tag-purple',
        details: 'Online Softmax 在线分块递推数学推导、双 WGMMA 异步重叠与 TMA 流水线编排。',
        keywords: ['Online Softmax', 'Dual-GEMM Overlap', 'FlashAttention-3', '740 TFLOPS']
      },
      {
        chapter: '27',
        title: '混合专家模型 (MoE) 分组 GEMM',
        link: '/workloads/moe_gemm',
        tag: 'MoE 算子',
        tagType: 'tag-blue',
        details: '动态 Token 路由负载倾斜痛点、单内核启动算完所有专家、前缀和与二分查找调度。',
        keywords: ['Grouped GEMM', 'Ragged Batch', 'Prefix Sum', 'Mixtral / DeepSeek']
      },
      {
        chapter: '28',
        title: '隐式 GEMM 卷积算子实现',
        link: '/workloads/conv2d_as_gemm',
        tag: '经典卷积',
        tagType: 'tag-green',
        details: '消灭 im2col 90% 显存膨胀、四维坐标 ALU 即时解码、Fprop/Dgrad/Wgrad 统一同构。',
        keywords: ['Implicit GEMM', '零 im2col 膨胀', '四维坐标解码', '谓词 Padding']
      },
      {
        chapter: '29',
        title: 'W4A16 / W8A8 混合精度量化算子',
        link: '/workloads/quantized_gemm',
        tag: '大模型量化',
        tagType: 'tag-amber',
        details: 'Decode 阶段访存瓶颈突破、IEEE-754 异或魔数极速反量化与指令交织延迟隐藏。',
        keywords: ['W4A16', 'Magic XOR', '实时反量化', 'AWQ / GPTQ']
      },
      {
        chapter: '30',
        title: 'LayerNorm / RMSNorm + GEMM 深度融合',
        link: '/workloads/layernorm_gemm_fusion',
        tag: '端到端融合',
        tagType: 'tag-purple',
        details: '跨算子显存往返彻底消除、Warp Shuffle 蝶形规约、共享内存驻留与推理时延削减 35%。',
        keywords: ['Fused Norm-GEMM', '蝶形网络规约', '跨算子零显存', '单内核完成整层']
      }
    ]
  },
  {
    title: '第六部分：权威参考与词典 (Reference & Glossary)',
    description: '全书统一技术词汇标准与形式化数学代数符号体系',
    articles: [
      {
        chapter: '31',
        title: '核心术语与代数符号权威词典',
        link: '/glossary',
        tag: '基准词典',
        tagType: 'tag-blue',
        details: '统一规范中英双语技术术语对照、形式化数学代数符号速查表，杜绝术语混乱。',
        keywords: ['Glossary', '术语统一', '数学符号', '快速检索']
      }
    ]
  }
]
</script>

<style scoped>
.doc-catalog-wrapper {
  margin: 2.5rem 0 4rem;
}

.catalog-section {
  margin-bottom: 3.5rem;
}

.catalog-section-header {
  border-bottom: 2px solid var(--vp-c-divider);
  padding-bottom: 0.85rem;
  margin-bottom: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

@media (min-width: 768px) {
  .catalog-section-header {
    flex-direction: row;
    align-items: flex-end;
    justify-content: space-between;
  }
}

.section-title-group {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.section-number-badge {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: 700;
  font-size: 0.8rem;
  padding: 0.2rem 0.6rem;
  border-radius: 6px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.section-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  letter-spacing: -0.01em;
}

.section-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.section-desc {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

.section-count-pill {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
  padding: 0.15rem 0.55rem;
  border-radius: 9999px;
  white-space: nowrap;
}

/* Grid */
.catalog-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1.25rem;
}

@media (min-width: 1200px) {
  .catalog-grid {
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  }
}

/* Card */
.catalog-card {
  display: flex;
  flex-direction: column;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 1.25rem;
  text-decoration: none !important;
  color: inherit !important;
  transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
  position: relative;
  overflow: hidden;
}

.catalog-card:hover {
  transform: translateY(-3px);
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 10px 24px -6px rgba(0, 0, 0, 0.1);
}

.catalog-card:hover .card-arrow-icon {
  transform: translateX(4px);
  color: var(--vp-c-brand-1);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.chapter-badge {
  font-family: var(--vp-font-family-mono);
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  padding: 0.15rem 0.5rem;
  border-radius: 5px;
}

.topic-tag {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 5px;
}

.tag-blue {
  background: rgba(37, 99, 235, 0.1);
  color: #2563eb;
}

.tag-purple {
  background: rgba(147, 51, 234, 0.1);
  color: #9333ea;
}

.tag-amber {
  background: rgba(217, 119, 6, 0.1);
  color: #d97706;
}

.tag-green {
  background: rgba(16, 185, 129, 0.1);
  color: #10b981;
}

.card-title {
  margin: 0 0 0.5rem;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  line-height: 1.4;
}

.card-arrow-icon {
  width: 16px;
  height: 16px;
  color: var(--vp-c-text-3);
  transition: transform 0.2s ease, color 0.2s ease;
  flex-shrink: 0;
}

.card-details {
  margin: 0 0 1rem;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  line-height: 1.55;
  flex-grow: 1;
}

.card-footer {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  padding-top: 0.75rem;
  border-top: 1px dashed var(--vp-c-divider);
}

.card-kw-pill {
  font-size: 0.725rem;
  color: var(--vp-c-text-3);
  background: var(--vp-c-bg-alt);
  padding: 0.1rem 0.45rem;
  border-radius: 4px;
}
</style>
