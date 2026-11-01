// SPDX-License-Identifier: AGPL-3.0-only
// The closest-point search of a deviation map on the GPU: one WebGPU thread
// per scan vertex walks the same packed BVH three-mesh-bvh built for the CPU
// path (its node buffer is uploaded as it is) and reports the nearest
// triangle and the squared distance. The sign — the pseudonormal of the
// feature the point lands on — is read on the CPU afterwards, once per point,
// so the exact rules of surface.ts stay in one place. Prototype: the plain
// query only; the facing-limited query of closestFacingWay stays on the CPU.
/// <reference types="@webgpu/types" />

import type { Rigid } from '../deviation/rigid'
import type { NominalSurface } from '../deviation/surface'

/** Threads per workgroup; a dispatch of ceil(points / 256) workgroups stays
 *  under the 65 535 limit up to 16.7 M points. */
const WORKGROUP = 256
/** Traversal stack depth; a SAH tree of 8-triangle leaves over 10 M triangles is about 25 deep. */
const STACK = 64

const SHADER = /* wgsl */ `
struct Params {
  r0: vec4<f32>, // rows of the rigid rotation, w unused
  r1: vec4<f32>,
  r2: vec4<f32>,
  t: vec4<f32>,  // translation, w = point count as f32 (bitcast below)
}
@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<u32>;
@group(0) @binding(3) var<storage, read> points: array<f32>;
@group(0) @binding(4) var<storage, read_write> outFace: array<u32>;
@group(0) @binding(5) var<storage, read_write> outD2: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

fn vertex(i: u32) -> vec3<f32> {
  return vec3<f32>(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
}

// Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection, 5.1.5).
fn triangleD2(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  let ab = b - a; let ac = c - a; let ap = p - a;
  let d1 = dot(ab, ap); let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return dot(ap, ap); }
  let bp = p - b;
  let d3 = dot(ab, bp); let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return dot(bp, bp); }
  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { let q = a + ab * (d1 / (d1 - d3)); let d = p - q; return dot(d, d); }
  let cp = p - c;
  let d5 = dot(ab, cp); let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return dot(cp, cp); }
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { let q = a + ac * (d2 / (d2 - d6)); let d = p - q; return dot(d, d); }
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) { let q = b + (c - b) * ((d4 - d3) / (d4 - d3 + d5 - d6)); let d = p - q; return dot(d, d); }
  let denom = 1.0 / (va + vb + vc);
  let q = a + ab * (vb * denom) + ac * (vc * denom);
  let d = p - q;
  return dot(d, d);
}

// Squared distance from p to the box of node n (n is the node's first u32).
fn boxD2(n: u32, p: vec3<f32>) -> f32 {
  let lo = vec3<f32>(bitcast<f32>(nodes[n]), bitcast<f32>(nodes[n + 1u]), bitcast<f32>(nodes[n + 2u]));
  let hi = vec3<f32>(bitcast<f32>(nodes[n + 3u]), bitcast<f32>(nodes[n + 4u]), bitcast<f32>(nodes[n + 5u]));
  let d = max(max(lo - p, p - hi), vec3<f32>(0.0));
  return dot(d, d);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let count = bitcast<u32>(params.t.w);
  if (i >= count) { return; }
  let s = vec3<f32>(points[i * 3u], points[i * 3u + 1u], points[i * 3u + 2u]);
  let p = vec3<f32>(dot(params.r0.xyz, s), dot(params.r1.xyz, s), dot(params.r2.xyz, s)) + params.t.xyz;

  var stack: array<u32, ${STACK}>;
  var top = 0u;
  var bestD2 = 3.4e38;
  var bestFace = 0xffffffffu;
  var n = 0u;
  var hasNode = true;
  loop {
    if (!hasNode) { break; }
    // Leaf: the high half of word 7 is 0xFFFF; the low half its triangle count.
    let w7 = nodes[n + 7u];
    if ((w7 >> 16u) == 0xffffu) {
      let first = nodes[n + 6u];
      let last = first + (w7 & 0xffffu);
      for (var f = first; f < last; f++) {
        let d2 = triangleD2(p, vertex(index[f * 3u]), vertex(index[f * 3u + 1u]), vertex(index[f * 3u + 2u]));
        if (d2 < bestD2) { bestD2 = d2; bestFace = f; }
      }
      hasNode = false;
    } else {
      let left = n + 8u;
      let right = nodes[n + 6u];
      let dl = boxD2(left, p);
      let dr = boxD2(right, p);
      // The nearer child first; the farther one waits on the stack, and is
      // dropped when it is already farther than the best triangle found.
      var near = left; var far = right; var dn = dl; var df = dr;
      if (dr < dl) { near = right; far = left; dn = dr; df = dl; }
      if (df < bestD2 && top < ${STACK}u) { stack[top] = far; top++; }
      if (dn < bestD2) { n = near; continue; }
      hasNode = false;
    }
    // Pop until a node that can still hold a closer point.
    loop {
      if (top == 0u) { break; }
      top--;
      let c = stack[top];
      if (boxD2(c, p) < bestD2) { n = c; hasNode = true; break; }
    }
  }
  outFace[i] = bestFace;
  outD2[i] = bestD2;
}
`

export interface GpuClosest {
  /** The nearest triangle of every point, and the squared distance to it; -1 (0xffffffff) where the search found nothing. */
  query(points: Float32Array, transform: Rigid): Promise<{ face: Uint32Array; d2: Float32Array; ms: { upload: number; compute: number; read: number } }>
  destroy(): void
}

/** The WebGPU device, or null where there is none (no adapter, or no WebGPU). */
export async function gpuDevice(): Promise<GPUDevice | null> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu
  if (!gpu) return null
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null)
  if (!adapter) return null
  // A scan-sized reference needs more than the 128 MB default per buffer.
  const want = adapter.limits.maxStorageBufferBindingSize
  return adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: want, maxBufferSize: adapter.limits.maxBufferSize } }).catch(() => null)
}

/** The surface's BVH and triangles on the device, ready to be queried by many point sets. */
export function gpuClosest(device: GPUDevice, surface: NominalSurface): GpuClosest {
  const roots = surface.packedBvh()
  if (roots.length !== 1) throw new Error('A reference with groups has several BVH roots; the GPU path takes one.')
  const upload = (data: ArrayBufferView | ArrayBuffer, usage: GPUBufferUsageFlags) => {
    const bytes = data instanceof ArrayBuffer ? data : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(bytes.byteLength / 4) * 4), usage: usage | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(b, 0, bytes)
    return b
  }
  const nodes = upload(roots[0], GPUBufferUsage.STORAGE)
  const positions = upload(surface.positions, GPUBufferUsage.STORAGE)
  const index = upload(surface.index, GPUBufferUsage.STORAGE)
  const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const module = device.createShaderModule({ code: SHADER })
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } })

  return {
    async query(points, transform) {
      const n = points.length / 3
      const t0 = performance.now()
      const pts = upload(points, GPUBufferUsage.STORAGE)
      const outFace = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
      const outD2 = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
      const readFace = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
      const readD2 = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
      const r = transform.r, t = transform.t
      const u = new Float32Array(16)
      u.set([r[0], r[1], r[2], 0, r[3], r[4], r[5], 0, r[6], r[7], r[8], 0, t[0], t[1], t[2]])
      new Uint32Array(u.buffer)[15] = n
      device.queue.writeBuffer(uniform, 0, u)
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [nodes, positions, index, pts, outFace, outD2, uniform].map((buffer, binding) => ({ binding, resource: { buffer } })),
      })
      const t1 = performance.now()
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bind)
      pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP))
      pass.end()
      enc.copyBufferToBuffer(outFace, 0, readFace, 0, n * 4)
      enc.copyBufferToBuffer(outD2, 0, readD2, 0, n * 4)
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      const t2 = performance.now()
      await Promise.all([readFace.mapAsync(GPUMapMode.READ), readD2.mapAsync(GPUMapMode.READ)])
      const face = new Uint32Array(readFace.getMappedRange().slice(0))
      const d2 = new Float32Array(readD2.getMappedRange().slice(0))
      const t3 = performance.now()
      for (const b of [pts, outFace, outD2, readFace, readD2]) b.destroy()
      return { face, d2, ms: { upload: t1 - t0, compute: t2 - t1, read: t3 - t2 } }
    },
    destroy() {
      for (const b of [nodes, positions, index, uniform]) b.destroy()
    },
  }
}

/** The deviation map of computeDeviation (plain query, no facing limit), with the search on the GPU. */
export async function gpuDeviation(closest: GpuClosest, surface: NominalSurface, scanPositions: Float32Array, transform: Rigid): Promise<{ values: Float32Array; d2: Float32Array; ms: { upload: number; compute: number; read: number; sign: number } }> {
  const { face, d2, ms } = await closest.query(scanPositions, transform)
  const t0 = performance.now()
  const values = surface.signOnFaces(scanPositions, transform, face)
  return { values, d2, ms: { ...ms, sign: performance.now() - t0 } }
}
