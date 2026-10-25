import DefaultTheme from 'vitepress/theme'
import WarningModal from './components/WarningModal.vue'
import HamburgerToggle from './components/HamburgerToggle.vue'
import './custom.css'
import { h } from 'vue'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-top': () => h(WarningModal),
      'nav-bar-content-before': () => h(HamburgerToggle)
    })
  }
}
