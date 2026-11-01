// SPDX-License-Identifier: AGPL-3.0-only
// The sample bracket as the reference, and as the scan its own vertices
// jittered by up to ±0.2 mm and duplicated to the point count asked for
// (?points=N). The CPU map (computeDeviation) is the truth; the GPU map must
// agree within float32 rounding, and the times say what the GPU buys.
import { parse3MF } from '../src/core/parsers/threemf'
import { weldTriangleSoup } from '../src/core/geometry/weld'
import { NominalSurface } from '../src/core/deviation/surface'
import { computeDeviation } from '../src/core/deviation/deviation'
import { identityRigid } from '../src/core/deviation/rigid'
import { gpuClosest, gpuDeviation, gpuDevice } from '../src/core/gpu/closestPoint'

const out = document.getElementById('out')!
const say = (line: string) => {
  out.textContent += `\n${line}`
  console.log(line)
}

async function run() {
  const wanted = Number(new URLSearchParams(location.search).get('points') ?? 1_000_000)
  const buffer = await (await fetch('/samples/fridgeBracket_01-reduced.3mf')).arrayBuffer()
  const mesh = parse3MF(buffer)
  const { positions, indices } = mesh.kind === 'indexed' && mesh.indices ? { positions: mesh.positions, indices: mesh.indices } : weldTriangleSoup(mesh.positions)
  let t = performance.now()
  const surface = new NominalSurface(positions, indices)
  say(`reference: ${surface.triangleCount} triangles, ${surface.vertexCount} vertices; BVH + normals ${(performance.now() - t).toFixed(0)} ms`)

  // The scan: vertices jittered, repeated to the count, deterministic.
  let seed = 12345
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1
  const n = wanted
  const scan = new Float32Array(n * 3)
  const vcount = surface.vertexCount
  for (let i = 0; i < n; i++) {
    const v = i % vcount
    scan[i * 3] = positions[v * 3] + rnd() * 0.2
    scan[i * 3 + 1] = positions[v * 3 + 1] + rnd() * 0.2
    scan[i * 3 + 2] = positions[v * 3 + 2] + rnd() * 0.2
  }
  const rigid = identityRigid()

  t = performance.now()
  const cpu = computeDeviation(surface, scan, rigid)
  const cpuMs = performance.now() - t
  say(`CPU computeDeviation: ${n} points in ${cpuMs.toFixed(0)} ms (one thread)`)

  const device = await gpuDevice()
  if (!device) {
    say('GPU: no WebGPU device')
    say('DONE')
    return
  }
  const info = (await (navigator.gpu.requestAdapter() as Promise<GPUAdapter>)).info
  say(`GPU: ${info.vendor} ${info.architecture} ${info.description}`)
  t = performance.now()
  const closest = gpuClosest(device, surface)
  say(`GPU upload of BVH + triangles: ${(performance.now() - t).toFixed(0)} ms`)
  // Warm once (pipeline compile), then time.
  await gpuDeviation(closest, surface, scan.subarray(0, 3 * 1024), rigid)
  t = performance.now()
  const gpu = await gpuDeviation(closest, surface, scan, rigid)
  const gpuMs = performance.now() - t
  say(`GPU deviation: ${gpuMs.toFixed(0)} ms (upload ${gpu.ms.upload.toFixed(0)}, compute ${gpu.ms.compute.toFixed(0)}, read ${gpu.ms.read.toFixed(0)}, sign on CPU ${gpu.ms.sign.toFixed(0)})`)

  let worst = 0, worstAt = -1, signFlips = 0, nan = 0
  for (let i = 0; i < n; i++) {
    const a = cpu[i], b = gpu.values[i]
    if (Number.isNaN(a) || Number.isNaN(b)) { nan++; continue }
    const d = Math.abs(a - b)
    if (d > worst) { worst = d; worstAt = i }
    if (Math.sign(a) !== Math.sign(b) && Math.abs(a) > 1e-4) signFlips++
  }
  say(`agreement: max |CPU−GPU| = ${worst.toExponential(2)} mm at point ${worstAt}; sign disagreements ${signFlips}; NaN ${nan}`)
  if (worstAt >= 0) say(`  at that point: CPU signed ${cpu[worstAt].toFixed(5)}, GPU signed ${gpu.values[worstAt].toFixed(5)}, GPU distance ${Math.sqrt(gpu.d2[worstAt]).toFixed(5)} — equal distances mean a tie between two surfaces`)
  say(`speedup vs one CPU thread: ${(cpuMs / gpuMs).toFixed(1)}×`)
  say('DONE')
  closest.destroy()
}

run().catch((e) => {
  say(`ERROR ${e?.stack ?? e}`)
  say('DONE')
})
