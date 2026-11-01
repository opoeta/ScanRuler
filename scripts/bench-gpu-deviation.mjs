// SPDX-License-Identifier: AGPL-3.0-only
// Runs bench/gpu-deviation.html in Electron on the DirectX 11 ANGLE backend —
// the one Chromium hands a WebGPU adapter out on under Windows — and prints
// what the page logs. Needs the dev server (npm run dev) and desktop/'s
// node_modules (electron).
//   node scripts/bench-gpu-deviation.mjs [points]
import { spawn } from 'node:child_process'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const electron = createRequire(join(root, 'desktop', 'package.json'))('electron')
const points = process.argv[2] ?? '1000000'
const url = `${process.env.APP_URL ?? 'http://localhost:5173/'}bench/gpu-deviation.html?points=${points}`

const dir = mkdtempSync(join(tmpdir(), 'scanruler-bench-'))
writeFileSync(join(dir, 'package.json'), JSON.stringify({ name: 'bench', main: 'main.cjs' }))
writeFileSync(join(dir, 'main.cjs'), `
const { app, BrowserWindow } = require('electron')
app.commandLine.appendSwitch('use-angle', 'd3d11')
app.commandLine.appendSwitch('enable-features', 'WebGPU')
app.commandLine.appendSwitch('enable-unsafe-webgpu')
app.commandLine.appendSwitch('ignore-gpu-blocklist')
app.whenReady().then(() => {
  const w = new BrowserWindow({ show: false, webPreferences: { sandbox: true } })
  w.webContents.on('console-message', (e) => { const line = e.message ?? e; console.log(line); if (String(line) === 'DONE') setTimeout(() => app.quit(), 100) })
  w.loadURL(${JSON.stringify(url)})
})
`)
const child = spawn(electron, [dir, `--user-data-dir=${join(dir, 'profile')}`], { stdio: ['ignore', 'inherit', 'ignore'] })
child.on('exit', (code) => process.exit(code ?? 0))
