/**
 * ESLint flat config for the whole repo: `npm run lint` runs `eslint .`.
 *
 * Standard recommended sets only (no stylistic rules), scoped by where code runs:
 * - src/ runs in the browser; src/worker/ in a Web Worker; src/engine/ in both a worker and Node
 *   (the eval harness), so it gets both global sets.
 * - React rules (hooks + fast-refresh) apply to the UI code only, not the engine or the worker.
 * - vite/vitest configs, scripts/ and eval/ run in Node.
 * Unused variables and parameters may be kept when prefixed with `_`, matching tsc's
 * noUnusedParameters convention.
 */

import js from '@eslint/js'
import { defineConfig, globalIgnores } from 'eslint/config'
import reactHooks from 'eslint-plugin-react-hooks'
import { reactRefresh } from 'eslint-plugin-react-refresh'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default defineConfig(
  globalIgnores(['dist', 'dev-dist', 'datasets', 'tools', 'coverage', 'node_modules']),
  {
    files: ['**/*.{js,mjs,cjs,ts,tsx}'],
    extends: [js.configs.recommended, tseslint.configs.recommended],
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrorsIgnorePattern: '^_' },
      ],
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: { globals: globals.browser },
  },
  {
    files: ['src/worker/**/*.ts'],
    languageOptions: { globals: globals.worker },
  },
  {
    files: ['src/engine/**/*.ts'],
    languageOptions: { globals: { ...globals.worker, ...globals.node } },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/engine/**', 'src/worker/**'],
    extends: [reactHooks.configs.flat.recommended, reactRefresh.configs.vite()],
  },
  {
    files: ['*.{js,ts}', 'scripts/**/*.ts', 'eval/**/*.ts'],
    languageOptions: { globals: globals.node },
  },
)
