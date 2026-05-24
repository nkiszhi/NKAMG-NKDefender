import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api-backend': {
        target: 'http://127.0.0.1:5005',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api-backend/, '')
      }
    }
  }
})
