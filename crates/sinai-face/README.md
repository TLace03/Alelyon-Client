# sinai-face

Sinai's face, as one crate that every window drawing Sinai links, so they all draw one Sinai. Alelyon's desktop
app (CENTCOM's Sinai page) and the Angel window both build on it. It draws nothing and opens no device: a window
compiles the shaders here and feeds them what the modules compute.

| Module | What it is |
|---|---|
| `body` | The baked bust (`assets/sinai_body.bin`, compiled in): its base shape, the 348 morph targets the creator drives and MakeHuman's 34 expression units, and the arithmetic a window runs whenever a shape changes. The weighted deltas are summed onto the base, the head is held still, the eyes, jaw hinge and lids are measured again on the new shape, and the normals and occlusion are recomputed the way the bake computed them. |
| `expression` | The tones Sinai's face takes (a smile, surprise, sadness and the rest), each as weights on the expression units, following the action units each expression is known by, with the lid, tilt and warmth that come with it, and a face that eases from one tone to the next. |
| `appearance` | How a person has shaped their Sinai: the creator's catalog (`assets/sinai_controls.json`: every control, the targets it drives and what its two ends were measured to do), the values set on it, the colours, a share code, and the file it is kept in. A saved file is read forgivingly: an unknown control is set aside with a notice and a value out of range is brought into range. |
| `state_home` | Where a window keeps that file (`~/.alelyon/angel`), and the one-time copy of the old `%APPDATA%` files into it: checked by SHA-256, never a move, and never made by an AI coding agent's process. |
| `shaders` | The WGSL that draws the bust (`ANGEL`), the full-screen copy of the offscreen frame (`BLIT`), the valley (`TERRAIN`) and the sky with its sun and stars (`SKY`). |

Sinai has no gender: no control shapes one, and the catalog says why the chest-muscle control stops at half its
travel.

## Use it

```toml
[dependencies]
sinai-face = { path = "crates/sinai-face" }
```

```rust
use sinai_face::{appearance::Catalog, body::Body, expression::Face};

let body = Body::load();                      // the baked bust, from the bytes compiled in
let catalog = Catalog::builtin();             // the creator's controls
let mut face = Face::new(&body);
face.ease_toward(&body, "smile", 0.2);        // part of the way towards a smile
let wgsl = sinai_face::shaders::ANGEL;        // compile with wgpu (naga) and draw
```

## Build and test

Rust 1.97 or later. The crate is its own workspace with a committed lockfile:

```bash
cargo test --locked
```

The tests need no window and no GPU. `body`'s are held to shapes the bake computed independently
(`src/testdata/body_golden.json`); one test, a timing rather than a check, is ignored by default and run by hand.

## Where the bust comes from

`assets/sinai_body.bin` is baked from the base mesh and morph targets of
[MakeHuman](https://github.com/makehumancommunity/makehuman). In September 2020 the MakeHuman project released
its assets under CC0 1.0: Data Collection AB, Joel Palmius and Jonas Hauquier, the copyright holders, waived their
rights worldwide, and section C of MakeHuman's `LICENSE.md` names the base mesh, the targets and the modifiers among
them. No attribution is required; it is given because it is deserved. The slider images MakeHuman ships beside the
targets are under the AGPL and were not taken.

The bake crops the mesh to a bust running from the crown to the lower ribs, ending each arm a little below the
shoulder, and records every vertex's distance from those cuts, so a window can dissolve the bust into light at its
edges. Two more weights per vertex drive breathing. Each control's label was decided by measuring what its two
ends do to the bust with the head held still, not by the target's file name, and the bake refuses to write a
catalog with a control that does not move the bust visibly or two controls that move it the same way. The bake
itself (a Python tool over the vendored MakeHuman files) is kept in Alelyon's source repository, not here.

## License

Licensed under the Apache License, Version 2.0. The baked bust is derived from MakeHuman's CC0 assets, as above.
