import { defineConfig } from 'vite'
import { tanstackStart } from '@tanstack/react-start/plugin/vite'
import viteReact from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  server: {
    port: 3001,
    proxy: {
      // fraudless-api (FastAPI) runs on :8001; the browser only ever talks to this origin
      '/api': {
        target: 'http://127.0.0.1:8001',
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  plugins: [tailwindcss(), tanstackStart(), viteReact()],
})
