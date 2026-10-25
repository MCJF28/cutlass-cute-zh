<template>
  <ClientOnly>
    <div v-if="showModal" class="warning-modal-backdrop" @click.self="dismiss">
      <div class="warning-modal-card" role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <div class="warning-modal-header">
          <div class="warning-icon-wrapper">
            <!-- Professional SVG Warning Triangle Icon -->
            <svg class="warning-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
          </div>
          <h2 id="modal-title" class="warning-title">访问须知与声明</h2>
        </div>

        <div class="warning-modal-body">
          <p class="warning-text-lead">
            本网站只是用来消耗反重力额度的，并且给有需要的人看一看，准确度极低。可用度极低，不想看的请不看，造成错误概不负责。
          </p>
          <div class="warning-subtext-box">
            <p class="warning-subtext">
              1. <strong>非官方文档</strong>：本站所有内容均由 AI 辅助生成与解读，非 NVIDIA 官方授权，亦非原厂出版物。<br/>
              2. <strong>权威溯源</strong>：涉及生产环境部署、PTX 汇编与架构参数，请务必以 NVIDIA CUTLASS 官方源码库为准。
            </p>
          </div>
        </div>

        <div class="warning-modal-footer">
          <button class="warning-confirm-btn" @click="dismiss">
            我已知晓并继续访问
          </button>
        </div>
      </div>
    </div>
  </ClientOnly>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const STORAGE_KEY = 'antigravity_quota_disclaimer_ack_v1'
const showModal = ref(false)

onMounted(() => {
  try {
    const acknowledged = localStorage.getItem(STORAGE_KEY)
    if (!acknowledged) {
      showModal.value = true
    }
  } catch (e) {
    showModal.value = true
  }
})

function dismiss() {
  showModal.value = false
  try {
    localStorage.setItem(STORAGE_KEY, 'true')
  } catch (e) {
    // Ignore storage quota errors in private browsing
  }
}
</script>

<style scoped>
.warning-modal-backdrop {
  position: fixed;
  inset: 0;
  z-index: 999999;
  background-color: rgba(0, 0, 0, 0.72);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
  animation: fadeIn 0.25s ease-out;
}

.warning-modal-card {
  width: 100%;
  max-width: 540px;
  background: var(--vp-c-bg, #ffffff);
  border: 1px solid var(--vp-c-danger-1, #e53e3e);
  border-radius: 16px;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(229, 62, 62, 0.2);
  padding: 2rem;
  color: var(--vp-c-text-1, #1a202c);
  transform: scale(1);
  animation: popIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.warning-modal-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.25rem;
}

.warning-icon-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background-color: rgba(229, 62, 62, 0.12);
  color: var(--vp-c-danger-1, #e53e3e);
  flex-shrink: 0;
}

.warning-svg {
  width: 28px;
  height: 28px;
}

.warning-title {
  font-size: 1.35rem;
  font-weight: 700;
  margin: 0;
  color: var(--vp-c-text-1, #1a202c);
  letter-spacing: -0.02em;
}

.warning-modal-body {
  margin-bottom: 1.75rem;
  font-size: 0.95rem;
  line-height: 1.65;
}

.warning-text-lead {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--vp-c-danger-1, #e53e3e);
  background-color: rgba(229, 62, 62, 0.08);
  border-left: 4px solid var(--vp-c-danger-1, #e53e3e);
  padding: 0.75rem 1rem;
  border-radius: 6px;
  margin: 0 0 1rem 0;
}

.warning-subtext-box {
  background: var(--vp-c-bg-alt, #f7fafc);
  border: 1px solid var(--vp-c-divider, #e2e8f0);
  border-radius: 8px;
  padding: 0.85rem 1rem;
}

.warning-subtext {
  font-size: 0.875rem;
  color: var(--vp-c-text-2, #4a5568);
  margin: 0;
  line-height: 1.55;
}

.warning-modal-footer {
  display: flex;
  justify-content: flex-end;
}

.warning-confirm-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.75rem 1.75rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: #ffffff;
  background: var(--vp-c-brand-1, #3b82f6);
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
}

.warning-confirm-btn:hover {
  background: var(--vp-c-brand-2, #2563eb);
  transform: translateY(-1px);
  box-shadow: 0 6px 12px -2px rgba(59, 130, 246, 0.4);
}

.warning-confirm-btn:active {
  transform: translateY(0);
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes popIn {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}
</style>
