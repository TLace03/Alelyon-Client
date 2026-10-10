# Sinai's simulator (CPU crates)

A physics engine and virtual environment in which every sense is a view of one world
state. These are its CPU crates: plain Rust, no GPU device, and each held to a reference
by its tests.

| Crate | What it is |
|---|---|
| `sim-contract` | The observation contract, v0: what the simulator hands over each tick (a bundle of the senses), what comes back (actions, scene resets), and the evaluation-only ground truth and perturbations that never reach the model. Plain data, serde JSON, `validate` methods, and committed JSON fixtures a Python training loop can read. |
| `sim-scene` | The one scene description that physics, rendering and the senses start from: bodies, joints, geoms, meshes, materials, renderable instances, cameras and actuators. SI units, every invariant validated, serde JSON that round-trips exactly. It imports a subset of MuJoCo's MJCF (`sim_scene::mjcf::load`), ported from MuJoCo's own compiler and held to MuJoCo's results by a parity test. |
| `sim-world` | The CPU side of the world state: the structure-of-arrays layout per environment slot, a host mirror that resets deterministically, the render view a renderer reads, the simulator's only random number generator (Philox4x32-10) and the step scheduler. |
| `sim-raycast` | The renderer's CPU ray caster: the scene description and its packing (a BVH per mesh, materials and instances), the pinhole camera, the conversion from the one scene description, and the host reference renderer that writes RGB, `f16` depth and segmentation frames on the processor. The device renderer, which runs the same algorithm in compute kernels, is not in this repository; Alelyon's app draws the simulator with this crate. |
| `sim-physics` | The CPU reference of the physics core: MuJoCo's reduced-coordinate dynamics, soft constraints (joint and tendon limits, friction loss, and contacts from eleven primitive colliders with frictionless, pyramidal and elliptic rows) solved by MuJoCo's Newton and conjugate-gradient solvers, with Euler and RK4 integration, generic over `f64` and `f32`. Ported from MuJoCo 3.14.0 and held to it by golden files, bit for bit in `f64` on the platform the goldens were recorded on. |

`sim-physics` and `sim-scene` carry MuJoCo's Apache-2.0 licence and a `NOTICE` naming
what was ported. Of the MJCF test models under `crates/sim-scene/tests/fixtures/mujoco`,
`humanoid.xml` is MuJoCo's own (its `NOTICE` names the commit); the others were written
for the simulator.

## Conventions

- SI units: metres, kilograms, seconds, radians, kelvin.
- World frame: right-handed, +Z up. A body's pose is relative to its parent.
- Quaternions are `[x, y, z, w]` (scalar last). MJCF's `quat` is MuJoCo's `[w, x, y, z]`;
  the importer converts it.
- Camera frame: the OpenCV frame, x right, y down, z forward.
- Colours in a scene are linear; colours authored in sRGB are converted on import.

## Build and test

Rust 1.97 or later. The lockfile is committed; build with `--locked`.

```bash
cd sim
cargo test --locked --workspace
```

The tests that regenerate the MuJoCo goldens need MuJoCo's Python package and the
generator scripts in Alelyon's source repository; outside it they print `SKIPPED` and
return. The goldens themselves are here, and every other test reads them.
