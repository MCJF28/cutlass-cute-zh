<template>
  <!-- Only render hamburger toggle on non-home documentation pages -->
  <div v-if="!isHome" class="desktop-hamburger-wrapper">
    <button 
      class="hamburger-btn" 
      :class="{ 'is-active': isDrawerOpen }"
      @click.stop.prevent="toggleDrawer"
      aria-label="切换文档导航菜单"
      title="切换文档导航菜单"
    >
      <div class="hamburger-icon">
        <span class="bar bar-1"></span>
        <span class="bar bar-2"></span>
        <span class="bar bar-3"></span>
      </div>
      <span class="hamburger-label">章节目录</span>
    </button>

    <!-- Drawer Backdrop Overlay -->
    <Teleport to="body">
      <div 
        v-if="isDrawerOpen" 
        class="custom-sidebar-backdrop"
        @click="closeDrawer"
      ></div>
    </Teleport>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRoute, useData } from 'vitepress'

const isDrawerOpen = ref(false)
const route = useRoute()
const { frontmatter } = useData()

// Detect whether current page is homepage
const isHome = computed(() => {
  return frontmatter.value?.layout === 'home' || route.path === '/' || route.path === '/index.html'
})

function toggleDrawer() {
  isDrawerOpen.value = !isDrawerOpen.value
  updateBodyClass()
}

function closeDrawer() {
  if (isDrawerOpen.value) {
    isDrawerOpen.value = false
    updateBodyClass()
  }
}

function updateBodyClass() {
  if (typeof document !== 'undefined') {
    if (isDrawerOpen.value) {
      document.documentElement.classList.add('custom-drawer-open')
    } else {
      document.documentElement.classList.remove('custom-drawer-open')
    }
  }
}

// Close drawer automatically when route changes
watch(() => route.path, () => {
  closeDrawer()
})

// Listen for Escape key to close drawer
function handleKeyDown(e) {
  if (e.key === 'Escape' && isDrawerOpen.value) {
    closeDrawer()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeyDown)
  if (typeof document !== 'undefined') {
    document.documentElement.classList.remove('custom-drawer-open')
  }
})
</script>

<style scoped>
.desktop-hamburger-wrapper {
  display: inline-flex;
  align-items: center;
  margin-right: 0.75rem;
}

.hamburger-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.35rem 0.75rem;
  border-radius: 8px;
  background: var(--vp-c-bg-alt, #f1f5f9);
  border: 1px solid var(--vp-c-divider, #e2e8f0);
  color: var(--vp-c-text-1, #1e293b);
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  transition: all 0.2s ease;
  user-select: none;
}

.hamburger-btn:hover {
  background: var(--vp-c-brand-soft, #eff6ff);
  border-color: var(--vp-c-brand-1, #3b82f6);
  color: var(--vp-c-brand-1, #3b82f6);
}

.hamburger-icon {
  width: 16px;
  height: 12px;
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.bar {
  display: block;
  width: 100%;
  height: 2px;
  background-color: currentColor;
  border-radius: 2px;
  transition: transform 0.2s ease, opacity 0.2s ease;
}

.hamburger-btn.is-active .bar-1 {
  transform: translateY(5px) rotate(45deg);
}

.hamburger-btn.is-active .bar-2 {
  opacity: 0;
}

.hamburger-btn.is-active .bar-3 {
  transform: translateY(-5px) rotate(-45deg);
}

.hamburger-label {
  font-size: 0.85rem;
  letter-spacing: -0.01em;
}

@media (max-width: 959px) {
  /* On mobile, VitePress has its own mobile menu */
  .desktop-hamburger-wrapper {
    display: none;
  }
}
</style>

<style>
/* Global styles for drawer operation */
.custom-sidebar-backdrop {
  position: fixed;
  inset: 0;
  z-index: 45;
  background-color: rgba(0, 0, 0, 0.45);
  backdrop-filter: blur(3px);
  animation: backdropFade 0.2s ease;
}

@keyframes backdropFade {
  from { opacity: 0; }
  to { opacity: 1; }
}

@media (min-width: 960px) {
  /* Override VitePress default sidebar to make it an elegant, clean drawer */
  .VPSidebar {
    top: 0 !important;
    bottom: 0 !important;
    left: 0 !important;
    width: 320px !important;
    max-width: 340px !important;
    padding-left: 20px !important;
    padding-right: 16px !important;
    padding-top: calc(var(--vp-nav-height) + 16px) !important;
    padding-bottom: 32px !important;
    transform: translateX(-100%) !important;
    transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.28s ease !important;
    z-index: 50 !important;
    box-shadow: none !important;
    border-right: 1px solid var(--vp-c-divider) !important;
    background-color: var(--vp-sidebar-bg-color, var(--vp-c-bg)) !important;
  }

  /* Slide in drawer when active */
  .custom-drawer-open .VPSidebar {
    transform: translateX(0) !important;
    box-shadow: 12px 0 35px -5px rgba(0, 0, 0, 0.25) !important;
  }

  /* Hide the VitePress sticky curtain element inside sidebar */
  .VPSidebar .curtain {
    display: none !important;
  }

  /* Reset content left padding so documentation is centered and spacious */
  .VPContent.has-sidebar {
    padding-left: 0 !important;
    transition: padding-left 0.25s ease;
  }

  .VPContent.has-sidebar .VPDoc {
    max-width: 1060px;
    margin: 0 auto;
  }
}

@media (min-width: 1440px) {
  /* Strictly enforce width and clean padding on wide/ultra-wide screens! */
  .VPSidebar {
    width: 320px !important;
    max-width: 340px !important;
    padding-left: 20px !important;
    padding-right: 16px !important;
  }
}
</style>
