import { defineConfig } from 'vitest/config'

// Engine tests run in plain Node: the engine is DOM-free by contract (see src/engine/types.ts).
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts', 'eval/**/*.test.ts'],
    environment: 'node',
    testTimeout: 30_000,
  },
})
