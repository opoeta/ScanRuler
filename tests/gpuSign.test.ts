// SPDX-License-Identifier: AGPL-3.0-only
// The GPU path hands back only the nearest triangle of each point; the sign
// and the exact distance are then read on the CPU by signOnFaces. Given the
// triangles closest() itself finds, that must reproduce computeDeviation.
import { describe, expect, it } from 'vitest'
import { computeDeviation } from '../src/core/deviation/deviation'
import { identityRigid, rigidFromQuaternion } from '../src/core/deviation/rigid'
import { NominalSurface, emptyHit } from '../src/core/deviation/surface'

function box(sx: number, sy: number, sz: number): { positions: Float32Array; indices: Uint32Array } {
  const p = [
    [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
    [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
  ].flat()
  const i = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
  return { positions: Float32Array.from(p), indices: Uint32Array.from(i) }
}

describe('signOnFaces', () => {
  it('reproduces computeDeviation from the nearest triangles', () => {
    const { positions, indices } = box(10, 6, 3)
    const surface = new NominalSurface(positions, indices)
    let seed = 7
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1
    const n = 5000
    const scan = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      scan[i * 3] = rnd() * 13
      scan[i * 3 + 1] = rnd() * 8
      scan[i * 3 + 2] = rnd() * 5
    }
    const transform = rigidFromQuaternion(0.99, 0.05, -0.08, 0.1)
    transform.t.set([0.3, -0.2, 0.1])
    const truth = computeDeviation(surface, scan, transform)
    // The faces the CPU query lands on, as the GPU kernel would report them.
    const face = new Uint32Array(n)
    const hit = emptyHit()
    const r = transform.r, t = transform.t
    for (let i = 0; i < n; i++) {
      const x = scan[i * 3], y = scan[i * 3 + 1], z = scan[i * 3 + 2]
      surface.closest(r[0] * x + r[1] * y + r[2] * z + t[0], r[3] * x + r[4] * y + r[5] * z + t[1], r[6] * x + r[7] * y + r[8] * z + t[2], hit)
      face[i] = hit.faceIndex
    }
    const values = surface.signOnFaces(scan, transform, face)
    for (let i = 0; i < n; i++) expect(values[i]).toBeCloseTo(truth[i], 9)
    // Inside points read negative, outside positive.
    const inside = computeDeviation(surface, Float32Array.from([0, 0, 0]), identityRigid())
    expect(inside[0]).toBeLessThan(0)
    expect(surface.signOnFaces(Float32Array.from([30, 0, 0]), identityRigid(), Uint32Array.from([0xffffffff]))[0]).toBeNaN()
  })

  it('exposes one packed BVH root for a plain mesh', () => {
    const { positions, indices } = box(1, 1, 1)
    const roots = new NominalSurface(positions, indices).packedBvh()
    expect(roots).toHaveLength(1)
    expect(roots[0].byteLength % 32).toBe(0)
  })
})
