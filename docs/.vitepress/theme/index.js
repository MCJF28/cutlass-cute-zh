import DefaultTheme from 'vitepress/theme'
import WarningModal from './components/WarningModal.vue'
import HamburgerToggle from './components/HamburgerToggle.vue'
import RoadmapGrid from './components/RoadmapGrid.vue'
import './custom.css'
import { h } from 'vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Register RoadmapGrid globally so it can be used cleanly in markdown
    app.component('RoadmapGrid', RoadmapGrid)
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-top': () => h(WarningModal),
      'nav-bar-content-before': () => h(HamburgerToggle)
    })
  }
}
