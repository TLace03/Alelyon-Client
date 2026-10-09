# sim-contract: the observation contract, v0

The schema of what crosses between Sinai's simulator and the Sinai side. Each
tick the simulator hands over one **bundle** per environment: every sense, read
from one world state at one tick. Sinai hands back **actions** and scene
**resets**. Two further types exist for evaluation only and are **never an input
to Sinai**: **ground truth** (the hidden causes behind a tick, and the
segmentation frames and what their ids mean) and **perturbations** (an
interference with one sense).

The design is in ADR-0042 of Alelyon's source repository, "The contract with
the Sinai session". Its v0 terms were set with the Sinai
session; any change to them is that session's call.

The crate is plain data plus `validate` methods. It depends on `serde` and
`serde_json` only, opens no device, and has no GPU code. The
Rust types are the source of truth; the JSON under [`fixtures/v0/`](fixtures/v0/)
is what a Python training loop reads, and a test regenerates and compares it on
every run.

## Contents

1. [Reading the fixtures from Python](#reading-the-fixtures-from-python)
2. [Conventions](#conventions)
3. [The bundle](#the-bundle)
4. [Channels](#channels)
5. [Actions and resets](#actions-and-resets)
6. [Evaluation only: ground truth and perturbations](#evaluation-only-ground-truth-and-perturbations)
7. [Rates](#rates)
8. [Validation and errors](#validation-and-errors)
9. [Fixtures](#fixtures)
10. [What v0 does not fix](#what-v0-does-not-fix)
11. [Build and test](#build-and-test)

## Reading the fixtures from Python

Every fixture is one JSON document, readable with the standard library.
[`fixtures/v0/index.json`](fixtures/v0/index.json) names the Rust type each file
holds and marks the evaluation-only ones.

```python
import json
from pathlib import Path

root = Path("crates/sim-contract/fixtures/v0")   # from the workspace folder
index = json.loads((root / "index.json").read_text())
assert index["schema_version"] == 0          # refuse a schema you do not know

bundle = json.loads((root / "bundle.json").read_text())
assert bundle["identity"]["schema_version"] == 0
rgb = bundle["rgb"]                          # None (null) if sight was not sampled
if rgb is not None:
    frame = rgb["value"]                     # a device reference, not pixels
    print(rgb["captured_at_s"], frame["buffer"], frame["width"], frame["height"])
```

A frame is a reference to a device buffer: pixels never pass through JSON.

## Conventions

| Quantity | Unit or convention |
|---|---|
| Time | seconds, simulated (`sim_time_s`, `captured_at_s`, `duration_s`, `seconds`) |
| Length, position | metres |
| Angle | radians; revolute joints are in radians and rad/s, prismatic joints in metres and m/s |
| Pressure, shear traction | pascals |
| Temperature | kelvin (absolute, so above 0) |
| Force | newtons |
| Smell concentration | mol/m^3 of gas at the nose |
| Taste concentration | mol/L of solution at the tongue contact |
| World frame | right-handed, +Z up |
| Camera frame | OpenCV: x right, y down, z forward along the view axis (a camera pose's quaternion rotates it into the world) |
| Intrinsics | pixels of the 448 x 448 frame, pixel centres at integer coordinates: the centre of the top-left pixel is (0, 0), a centred principal point is `((W - 1) / 2, (H - 1) / 2)` = (223.5, 223.5). The v0 bundle carries no intrinsics; where they are stated they use this convention |
| Depth | z in the camera frame, along the view axis (not along the pixel's ray), in metres; a pixel whose ray hits nothing is `+inf` (`f16` bits `0x7C00`) |
| RGB | sRGB-encoded `u8` (the sRGB transfer curve applied, not linear light) |
| Segmentation ids | `u16` instance id per pixel, stable for a whole episode, 0 is background, meaning only through the per-episode seg table (evaluation only: ground truth, never the bundle) |
| Quaternion | `[x, y, z, w]` (scalar last), length 1 within 1e-3 |
| Identifiers (`scene`, `state_delta_id`) | 1 to 128 characters from `[a-z0-9_.-]`, starting with a letter or digit |
| Numbers | finite: no NaN, no infinity (JSON cannot carry them; `serde_json` writes `null`, which does not read back). This is about the numbers in the contract's types: a depth pixel on the device may be `+inf` |

JSON uses the Rust field names, in `snake_case`; unit-only enums are lowercase
strings (`"sight"`, `"u8"`, `"depth_metres"`); the enums with data carry a
`kind` tag (`{"kind": "look_at", ...}`).

## The bundle

`Bundle` is the observation for one environment at one tick. **Every channel is
always present**:

- an optional channel set to `null` means **not sampled this tick** (the senses
  sample at different rates; see [Rates](#rates));
- an empty value means **the channel exists and has no content yet** (no audio
  samples, no joints, no sensors, an empty species vector). Phase 1 needs every
  channel in the schema from day one, so the loop and the schema are whole before
  content arrives.

The JSON of a bundle has all eight keys. A reader that finds one missing has found
an error: the Rust reader refuses it.

| Key | Type | Sampled | Notes |
|---|---|---|---|
| `identity` | `Identity` | every tick | schema version, environment, episode, step, simulated time, seed |
| `rgb` | `Capture<FrameRef>` or `null` | 10 Hz | 448 x 448, `u8` x 3, on the device |
| `depth` | `Capture<FrameRef>` or `null` | with `rgb` | 448 x 448, `f16` metres, only when `rgb` is present (an optional sense channel) |
| `audio` | `Capture<AudioChunk>` | every tick, continuous | 16 kHz mono `f32` |
| `proprioception` | `Capture<Proprio>` | every tick (the control rate) | joints and optional base pose |
| `touch` | `Capture<TouchMaps>` or `null` | 100 Hz | one map per skin sensor |
| `smell` | `Capture<SpeciesVector>` or `null` | 10 Hz | mol/m^3, indexed by a species table |
| `taste` | `Capture<SpeciesVector>` or `null` | 10 Hz | mol/L, indexed by a species table |

**Segmentation is not in the bundle.** Per-instance ids that persist through
occlusion would hand the model the object identities that the object-permanence
test measures, so segmentation is evaluation only and lives in `GroundTruth`
(see [Evaluation only](#evaluation-only-ground-truth-and-perturbations)).
2026-10-01: segmentation moved from Bundle to GroundTruth by the contract owner.

**`Identity`**: `schema_version: u32` (0 in this version; check it first),
`env_id: u32`, `episode_id: u64`, `step: u64` (control ticks since the episode
began), `sim_time_s: f64`, `seed: u64` (the episode's seed). The same seed on the
same device and driver reproduces the episode bit for bit.

**`Capture<T>`**: `{ "captured_at_s": f64, "value": T }`. Each sense carries its
own capture time, because the senses sample at different rates. For an
unperturbed bundle it is at or before `identity.sim_time_s` and later than the
previous tick.

## Channels

**`FrameRef`** (sight). A frame that stays on the device, so the training loop
reads it from the buffer the renderer wrote with no round trip through the host.
Fields: `buffer` (`DeviceBufferId`, an opaque handle meaningful only inside the
process and device that issued it), `byte_offset`, `width`, `height`, `channels`,
`dtype` (`"u8" | "f16" | "u16" | "f32"`), `row_stride_bytes`, `semantic`
(`"rgb" | "depth_metres" | "segmentation"`). Row `y` starts at
`byte_offset + y * row_stride_bytes`. Rows may be padded but not overlap, and
offset and stride are multiples of the element size. The v0 formats, all
448 x 448: RGB `u8` x 3, depth `f16` metres x 1, segmentation `u16` x 1. Depth is
pixel-aligned with the RGB frame and is sent only with it, in the bundle.
Segmentation is pixel-aligned with the RGB frame of the same camera and tick but
is evaluation only: it is carried by `GroundTruth`, never by a bundle.
`FrameRef::v0_rgb()`, `v0_depth()` and `v0_segmentation()` return the v0 formats
(place them with `.at(buffer, byte_offset)`).

**Camera conventions** (fixed by the contract owner; the table in
[Conventions](#conventions) has them too):

- the camera frame is OpenCV's: x right, y down, z forward along the view axis;
- intrinsics are in pixels of the 448 x 448 frame, with pixel centres at integer
  coordinates, so a centred principal point is `((W - 1) / 2, (H - 1) / 2)`;
- depth is z along the view axis in metres, and a pixel with no hit is `+inf`
  (`f16` `0x7C00`);
- RGB is sRGB-encoded `u8`.

**`AudioChunk`** (sound). `sample_rate_hz` (16,000 in v0), `first_sample_index`
(position in the whole stream since the episode began) and `samples`, either
`{"kind": "host", "values": [f32...]}` (mono) or
`{"kind": "device", "buffer", "byte_offset", "count"}` (`f32` samples that stay on
the device). **Continuity**: chunk n+1's `first_sample_index` equals chunk n's
plus its sample count, with no gap and no overlap. `AudioContinuity::check(prev,
next)` verifies it. A chunk may be empty and still has its place in the stream.

**`Proprio`** (proprioception). `joint_positions` and `joint_velocities`, equal
length lists in the articulation's joint order, and an optional `base_pose`
(`position`, `orientation_quat`) in the world frame for a body whose base moves.

**`TouchMaps`** (touch). `sensors`: a list of per-sensor maps `{ sensor_id, rows,
cols, pressure_pa, shear_pa, temperature_k }`, row-major (taxel at row `r`, column
`c` is entry `r * cols + c`), each list exactly `rows * cols` long. `shear_pa`
entries are `[x, y]` in the sensor's frame. Pressure is not negative; a taxel not
in contact reads 0 Pa and the skin's temperature there. Sensor ids are unique.
No sensors is valid.

**`SpeciesTable`, `SpeciesVector`** (smell and taste). Percepts are **indexed by
PubChem CID, never by words**: mapping molecules to percepts is the Sinai side's
model, fed by these vectors. A `SpeciesTable` is `{ version, species: [{
pubchem_cid, name }] }`; a `SpeciesVector` is `{ table_version, concentrations }`,
one concentration per species of that table version, in table order. A table
version covers contents and order, so adding, removing or reordering a species is
a new version. Validation refuses duplicate CIDs, a CID of 0, a version mismatch,
a length mismatch, negative or non-finite concentrations, and an empty table for
a non-empty vector. `name` is a label for a person (the molecule's name); nothing
keys on it. The same table type serves both channels; the units differ by
channel (above).

## Actions and resets

What Sinai sends back. These are read **strictly**: an unknown field is refused,
so a misspelt field cannot silently do nothing. `Action::from_json`,
`Reset::from_json` and `Perturbation::from_json` read and validate in one step.

**`Action`**: `identity_step` (the `step` of the bundle it answers; the simulator
applies it to the tick that follows), `joint_targets` (radians or metres, joint
order), and `sensor`: `SensorActions`, whose two fields may be omitted:

- `camera`: `{"kind": "pose", "position", "orientation_quat"}` or
  `{"kind": "look_at", "target", "up"}` (`up` is not the zero vector).
- `sniff`: `{ "intensity": 0..=1, "duration_s": > 0 }`, which drives the sniff
  dynamics. Active perception is part of Sinai's design.

**`Reset`**: `{ "env_id", "seed", "scene" }`. Starts a new episode of `env_id`
from `scene` (a `SceneId`, an identifier: it can reach a path on disk, so it
cannot contain a separator or a leading dot) seeded with `seed`.

## Evaluation only: ground truth and perturbations

**These two types never reach a model.** A model shown the temperature at a point,
the progress of a reaction, a contact event or the identity of an occluded object
has been handed the answer to what its senses are meant to let it infer. Training
on it, or letting it into a model's context at evaluation time, would make every
cross-sense result meaningless. Segmentation is on this list: a segmentation
frame's per-instance ids persist through occlusion, so they are the identities
of the objects the model is asked to keep track of, and so is the table that says
what the ids mean. They exist so a test can compare what the model inferred with
what was true, and so the registered cross-sense tests (cue conflict and binding)
can make the senses disagree on purpose.

They are separate types, and no field of a `Bundle` can hold them, directly or
through any type a bundle holds. The test `tests/evaluation_only.rs` fails to
compile if a field is added to a bundle or to anything in it (so a segmentation
field cannot come back by accident), refuses `SegFrame`, `SegTable` and
`ShownSegFrame` as observations, and checks that no bundle JSON has a
segmentation key or frame. The index (`fixtures/v0/index.json`) marks their
fixtures `evaluation_only: true`, and so `frame_ref_segmentation.json`; a Python
training loop must not read them as input.

**`GroundTruth`**: `identity`, `temperatures_k` (named points, kelvin),
`reaction_progress` (named reactions such as `maillard` and `water_loss`, a
fraction in `0..=1`), `contacts` (`a`, `b`, `normal_force_n`, `position`) and
`object_ids_through_occlusion` (a list of `[entity_id, visible]` pairs),
`segmentation` (the **true** segmentation frames, at most one per camera),
`seg_table` (the per-episode id table, or `null`) and `shown_segmentation` (the
frames shown under a sight perturbation). All three are described below. Every
key is always written (`null` for a `seg_table` this record does not carry), and
a record with one left out is an error to read. Names are unique within a list.

2026-10-01: one SegFrame per camera, the per-episode seg table, and shown-vs-true
segmentation under perturbation, decided by the contract owner.
The table's `episode_id` must equal the record's `identity.episode_id`. A record
holds at most one shown frame per camera, for the same reason as the true ones:
the evaluator could not tell which shown frame Sinai saw.

**`SegFrame`** (an element of `GroundTruth.segmentation`): `{ "camera": u32,
"frame": Capture<FrameRef> }`. `camera` is the index of the camera that rendered
the frame; `frame` is a v0 segmentation frame (`FrameRef::v0_segmentation()`:
`u16` x 1, 448 x 448) with its capture time, pixel-aligned with the RGB frame of
the same camera and tick. **Ids are stable for a whole episode** (an object keeps
its id through occlusion and motion) **and 0 is background.** That is a property
the simulator must keep: the crate validates one frame's format and capture time
(a failure is reported as `Channel { channel: "segmentation", .. }`) and cannot
check ids across ticks. **A record has at most one true `SegFrame` per camera**:
a second frame for the same camera is refused as `Channel { channel:
"segmentation", cause: DuplicateId { field: "camera", id } }`. `segmentation`
**always describes the true state**, perturbed or not.

**`SegTable`** (`GroundTruth.seg_table`): `{ "episode_id": u64, "entries":
[[seg_id, entity_id], ...] }`, what each segmentation id means. It is written at
reset and fixed for the episode, so the record written at reset carries it
(`Some`) and later records may leave it `null` and be read against it. Seg id 0 is
background and **never appears** in the table, seg ids are unique, and entity ids
are unique (refusals: `OutOfRange` or `DuplicateId` on `seg_table.seg_id` or
`seg_table.entity_id`). **The ids are assigned, not derived by any formula** from
the entity, its kind or its position, so nothing about identity can be decoded
from an id without the table, and **the table lives only in the evaluation
channel**. Its `episode_id` must be the record's `identity.episode_id` (otherwise
`EpisodeMismatch { field: "seg_table.episode_id", expected, found }`). That the
table stays the same for the whole episode and that no formula is behind the ids
are properties the simulator must keep; the crate checks one table.

**`ShownSegFrame`** (an element of `GroundTruth.shown_segmentation`): `{
"perturbation": Perturbation, "frame": SegFrame }`, the segmentation of the frame
actually **shown** while a sight perturbation is active, with that perturbation.
It goes **beside** the true `SegFrame` of the same camera and never instead of
it, so an evaluation can compare what was shown with what was true. Each shown
frame's camera must have a true `SegFrame` in the same record
(`Channel { channel: "shown_segmentation", cause: ShownWithoutTrueSegmentation {
camera } }`), and its perturbation must be a sight one (`Offset`, `Delay` or
`RenderFromPerturbedState` with `sense` `"sight"`); the perturbation and the
frame are also validated as themselves. **At most one shown frame per camera**: a
second one for the same camera is refused as `Channel { channel:
"shown_segmentation", cause: DuplicateId { field: "camera", id } }`. No sight
perturbation active means an empty list.

**`Perturbation`** (one sense per perturbation; `Sense` is `"sight" | "sound" |
"touch" | "smell" | "taste" | "proprioception"`; sight's perturbation covers RGB
and depth together, and never the true segmentation in `GroundTruth`: the
segmentation of what was shown is recorded beside it, see `ShownSegFrame`):

- `{"kind": "offset", "sense", "seconds"}`: shift the sense's capture time
  (signed: positive later, negative earlier); the stamp and the content move.
- `{"kind": "delay", "sense", "seconds"}`: report what the world was `seconds`
  ago while the other senses report now (`seconds` is not negative).
- `{"kind": "render_from_perturbed_state", "sense", "state_delta_id"}`: render
  the sense from a registered perturbed copy of the world state.

Applying a perturbation is the simulator's job. This crate defines and validates
the request.

## Rates

`RATES_V0`: RGB (with depth) 10 Hz, audio 16 kHz continuous,
touch 100 Hz, smell 10 Hz, taste 10 Hz. Proprioception runs at the control rate,
which is a parameter. `Rates::due(sense, step, control_hz)` says whether a sense is
sampled at a step. It is integer arithmetic, a pure function with no clock:

- audio and proprioception are due at every step;
- sample `k` of a sense at `rate` Hz is taken at `k / rate` seconds and delivered
  with the first step at or after it, so the sense is due at step `n` when
  `floor(n * rate / control_hz)` exceeds its value at step `n - 1`; step 0 is
  always due;
- a sense whose rate is at least the control rate is due at every step (a bundle
  carries one touch map per tick, not every 100 Hz sample).

At 50 Hz control, RGB, smell and taste are due at steps 0, 5, 10, ...; at 25 Hz
control, RGB is due at steps 0, 3, 5, 8, 10, ... (0.04 s steps; samples at 0, 0.1,
0.2 s, ...).

## Validation and errors

Every type has a `validate` (those indexed by a table take it as an argument:
`SpeciesVector::validate(&table)`, `Bundle::validate(&smell_table, &taste_table)`).
Each returns `Result<(), ContractError>`. `ContractError` is data, so a caller can
match the rule that fired; a failure inside a bundle's channel is wrapped with the
channel's name, and `ContractError::root` reaches the rule. Producers validate
before handing a value over; consumers may validate what they receive. Nothing in
the crate panics on a bad value.

## Fixtures

[`fixtures/v0/`](fixtures/v0/) holds one JSON file per type (one per variant for
the tagged enums) and `index.json`. `tests/fixtures.rs` rebuilds every value in
code, writes it the way the files are written, and compares the bytes; it then
reads each committed file back and validates it. After an intended change,
regenerate from the workspace folder and review the diff:

```text
SIM_CONTRACT_REGENERATE_FIXTURES=1 cargo test --locked -p sim-contract --test fixtures
```

`index.json` marks `frame_ref_segmentation.json` evaluation-only along with the
ground-truth and perturbation fixtures, because a segmentation frame is ground
truth (the bundle fixtures hold none).

The fixtures are examples of the schema's shape, not simulator output: buffer ids
are placeholders and the concentrations are made up. The species are examples
too, but their PubChem CIDs are real. On 2026-10-01 PubChem's PUG REST service
(`compound/cid/<cids>/property/Title`) returned these titles for the six CIDs:

| CID | PubChem title |
|---|---|
| 702 | Ethanol |
| 176 | Acetic Acid |
| 6184 | Hexanal |
| 5234 | Sodium Chloride |
| 5988 | Sucrose |
| 33032 | L-Glutamic Acid |

A `.gitattributes` keeps their line endings LF.

## What v0 does not fix

These are open, and are the Sinai session's to settle. The crate does not decide
them silently.

- **Segmentation ids across ticks.** That ids are stable for a whole episode, that
  0 is background, that the seg table is fixed for the episode and that the ids
  are not derived by any formula are stated, but only the simulator can keep
  them: the crate checks one frame's format and one table, not one tick against
  another.
- **Quaternion order.** `[x, y, z, w]` (scalar last) is this crate's choice for
  the JSON; MuJoCo orders `[w, x, y, z]`. Confirm before the loop reads quaternions.
- **Audio level and chunk boundaries.** The sample type and rate are fixed; the
  amplitude scale is not, and neither is how many samples a tick's chunk holds
  (continuity is what is fixed).
- **Joint order and count.** The articulation's joint order is the simulator's;
  the contract carries the lists and requires positions and velocities to agree.
- **`Action.identity_step`.** Documented as the `step` of the bundle the action
  answers, applied to the following tick.
- **Reaction progress as a fraction.** `reaction_progress` is constrained to
  `0..=1`; a quantity in other units (grams of water lost) would be a different
  channel.
- **Offset and delay semantics.** The ranges are validated here; exactly how the
  simulator applies them is the simulator's, to be pinned when it is built.

## Build and test

From the workspace folder, two levels up:

```text
cargo test --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

The tests: `tests/contract.rs` (round trips, every refusal, the rates rule),
`tests/evaluation_only.rs` (ground truth, segmentation frames, the seg table, shown
frames and perturbations cannot reach a bundle),
and `tests/fixtures.rs` (the committed JSON matches the code). No test opens a GPU
device.
