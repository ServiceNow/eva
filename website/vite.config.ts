import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import yaml from 'js-yaml'

const JUDGE_PROMPTS_ID = 'virtual:judge-prompts'
const JUDGE_YAML = fileURLToPath(new URL('../configs/prompts/judge.yaml', import.meta.url))

// Exposes the judge prompts from configs/prompts/judge.yaml, with placeholders
// left as-is.
function judgePrompts(): Plugin {
  return {
    name: 'judge-prompts',
    resolveId(id) {
      if (id === JUDGE_PROMPTS_ID) return '\0' + JUDGE_PROMPTS_ID
    },
    load(id) {
      if (id !== '\0' + JUDGE_PROMPTS_ID) return
      this.addWatchFile(JUDGE_YAML)
      const { judge } = yaml.load(readFileSync(JUDGE_YAML, 'utf8')) as {
        judge: Record<string, Record<string, string>>
      }
      return `export default ${JSON.stringify(judge)}`
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), judgePrompts()],
  base: './',
  build: {
    outDir: '../docs',
    emptyOutDir: false,
  },
})
