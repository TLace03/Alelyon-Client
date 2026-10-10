# sim-raycast: the simulator's CPU ray caster

The renderer of Sinai's simulator, on the processor. It holds what the renderer
draws and how, with no graphics device and no GPU dependency:

| Module | What it is |
|---|---|
| `scene` | The scene description (meshes, materials, instances, lighting) and its packing into flat records: a BVH per mesh, materials and instances |
| `from_scene` | The one scene description (`sim_scene::Scene`) converted into a renderer scene: geoms tessellated, imported meshes flat-shaded |
| `reference` | The ray caster: one primary ray per pixel through the posed instances, physically based shading under a sun, a sky and a ground, the ACES tone curve and sRGB |
| `camera` | The pinhole camera and its 64-byte record |
| `layout` | The byte layouts of the packed records and of the output frames (RGB, `f16` depth, `u16` segmentation, as the observation contract defines them) |
| `scenes` | Built-in scenes for benchmarks and tests |
| `bvh`, `mesh`, `math`, `f16` | The helpers |

The device renderer (`sim-render`, on a compute device) runs the same algorithm
in compute kernels and re-exports these modules under its own paths, so a device
frame can be compared with this one pixel by pixel. Alelyon's desktop app draws
the simulator's Compute page with it on the processor alone.

```bash
cargo test --locked -p sim-raycast
```
