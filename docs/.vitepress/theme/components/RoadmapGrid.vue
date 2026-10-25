<template>
  <div class="roadmap-container">
    <div class="roadmap-grid">
      <!-- 阶段 1 -->
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

      <!-- 阶段 2 -->
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

      <!-- 阶段 3 -->
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

      <!-- 阶段 4 -->
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
</template>

<script setup>
</script>
