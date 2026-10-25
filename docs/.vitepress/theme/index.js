import DefaultTheme from 'vitepress/theme'
import WarningModal from './components/WarningModal.vue'
import HamburgerToggle from './components/HamburgerToggle.vue'
import RoadmapGrid from './components/RoadmapGrid.vue'
import DocIndexCatalog from './components/DocIndexCatalog.vue'
import './custom.css'
import { h } from 'vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Register custom components globally
    app.component('RoadmapGrid', RoadmapGrid)
    app.component('DocIndexCatalog', DocIndexCatalog)
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-top': () => h(WarningModal),
      'nav-bar-title-before': () => h(HamburgerToggle)
    })
  }
}
