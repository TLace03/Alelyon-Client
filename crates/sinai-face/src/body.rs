//! Sinai's bust, and every shape it can take.
//!
//! The head this replaces was a fixed mesh: baked once, uploaded once, posed
//! only by two hinges in the shader. A person shaping Sinai changes the mesh
//! itself, so this module holds the base shape and every morph target the
//! creator offers, and does the arithmetic the window needs whenever any of
//! them moves: the weighted deltas summed onto the base, the head held still,
//! the eyes, jaw hinge and eyelids measured again on the new shape, and the
//! normals and occlusion recomputed the way the bake computed them.
//!
//! Everything here is plain arithmetic on vectors. It draws nothing and opens
//! no device, so it is tested without a window or a GPU, against shapes the
//! Python bake computed independently (`src/testdata/body_golden.json`).
//!
//! The file is `assets/sinai_body.bin`, baked from the MakeHuman base mesh and
//! targets (CC0; the crate's README says where they come from and how the
//! bake works).

use std::collections::HashMap;

pub const MAGIC: &[u8; 8] = b"SINAIBD1";
const BAKED: &[u8] = include_bytes!("../assets/sinai_body.bin");

/// Per vertex, as the GPU reads it: position (3), normal (3), then occlusion,
/// jaw weight, lid weight and part as one vec4, then the fade towards a cut,
/// the chest's swell, the shoulders' lift and the creator's highlight as
/// another.
pub const FLOATS_PER_VERTEX: usize = 14;
pub const STRIDE: u64 = (FLOATS_PER_VERTEX * 4) as u64;

// Part ids, as the bake writes them into each vertex. The shader keeps its own
// copies (angel.wgsl); here only the tests read them.
#[cfg(test)]
pub const PART_SKIN: f32 = 0.0;
#[cfg(test)]
pub const PART_BALL: f32 = 1.0;
#[cfg(test)]
pub const PART_TEETH_U: f32 = 2.0;
#[cfg(test)]
pub const PART_TEETH_L: f32 = 3.0;

/// The header, in bytes: magic, eight counts, four floats, two eye ranges.
const HEADER: usize = 8 + 8 * 4 + 4 * 4 + 2 * 2 * 4;

/// One morph target: which vertices it moves and by how much at full weight.
pub struct Target {
    pub name: String,
    pub idx: Vec<u32>,
    pub delta: Vec<[f32; 3]>,
}

/// Where each measurement on the face is taken from. Indices into the full
/// vertex list, which runs on past the drawn vertices into ones that exist
/// only to be measured.
struct Landmarks {
    eye_centre: [Vec<u32>; 2],
    eye_size: [Vec<u32>; 2],
    ears: Vec<u32>,
    anchor: Vec<u32>,
}

/// What the shader needs to pose a shape: each eye's centre and radius, Sinai's
/// left first, and the jaw hinge's height and depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rig {
    pub eyes: [[f32; 4]; 2],
    pub hinge: [f32; 2],
}

/// A shape, ready to draw.
pub struct Posed {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub ao: Vec<f32>,
    pub rig: Rig,
}

pub struct Body {
    pub n_draw: usize,
    /// Drawn vertices, then the landmark-only ones.
    base: Vec<[f32; 3]>,
    /// The normals the bake computed; the window recomputes its own, and the
    /// tests hold the two equal.
    #[cfg_attr(not(test), allow(dead_code))]
    base_normals: Vec<[f32; 3]>,
    /// Occlusion, jaw, lid, part, fade, chest, lift and a spare, per drawn vertex.
    attrs: Vec<[f32; 8]>,
    pub edges: Vec<u32>,
    pub tris: Vec<u32>,
    sphere: Vec<[f32; 3]>,
    eye_ranges: [(usize, usize); 2],
    ao_scale: f64,
    hinge_offset: [f64; 2],
    landmarks: Landmarks,
    targets: Vec<Target>,
    by_name: HashMap<String, usize>,
    anchor0: [f64; 3],
}

/// Reads little-endian values off the front of a byte slice, refusing to run
/// past its end. Every count in the file is checked against what is left
/// before anything is allocated for it.
struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).ok_or("a count overflows")?;
        if end > self.b.len() {
            return Err(format!(
                "the file ends at byte {} but needs {} more at byte {}",
                self.b.len(),
                n,
                self.at
            ));
        }
        let out = &self.b[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u16(&mut self) -> Result<u16, String> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    fn u32(&mut self) -> Result<u32, String> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn count(&mut self, what: &str, size: usize) -> Result<usize, String> {
        let n = self.u32()? as usize;
        if n.checked_mul(size).map_or(true, |bytes| bytes > self.b.len() - self.at) {
            return Err(format!("{what}: {n} entries cannot fit in what is left of the file"));
        }
        Ok(n)
    }

    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, String> {
        let s = self.take(n.checked_mul(4).ok_or("a count overflows")?)?;
        Ok(s.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn vec3s(&mut self, n: usize) -> Result<Vec<[f32; 3]>, String> {
        let s = self.take(n.checked_mul(12).ok_or("a count overflows")?)?;
        Ok(s.chunks_exact(12)
            .map(|c| {
                let f = |k: usize| f32::from_le_bytes([c[k], c[k + 1], c[k + 2], c[k + 3]]);
                [f(0), f(4), f(8)]
            })
            .collect())
    }

    fn name(&mut self) -> Result<String, String> {
        let n = self.u16()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "a name that is not UTF-8".to_string())
    }
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn wide(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn narrow(p: [f64; 3]) -> [f32; 3] {
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

/// The bust, read once for the life of the process: a few megabytes of shapes
/// that never change while it runs.
pub fn shared() -> &'static Body {
    static BODY: std::sync::OnceLock<Body> = std::sync::OnceLock::new();
    BODY.get_or_init(Body::load)
}

impl Body {
    /// The bust that ships inside the binary. Panics on a bad file rather than
    /// drawing something wrong: this file is compiled in, so a failure here is
    /// a build mistake rather than anything a person can cause.
    pub fn load() -> Body {
        match Body::parse(BAKED) {
            Ok(body) => body,
            Err(why) => panic!("the baked bust is unreadable: {why}"),
        }
    }

    pub fn parse(bytes: &[u8]) -> Result<Body, String> {
        if bytes.len() < HEADER || &bytes[0..8] != MAGIC {
            return Err("not a baked bust (wrong magic or too short)".into());
        }
        let mut r = Reader { b: bytes, at: 8 };
        let mut counts = [0usize; 8];
        for c in counts.iter_mut() {
            *c = r.u32()? as usize;
        }
        let [n_draw, n_extra, n_edge_idx, n_tri_idx, n_sphere, n_landmarks, n_targets, _] = counts;
        let ao_scale = r.f32()? as f64;
        let hinge_offset = [r.f32()? as f64, r.f32()? as f64];
        // The width the fade was measured over: already in every vertex's
        // fade, so it is read past rather than kept.
        let _fade_band = r.f32()?;
        let mut eye_ranges = [(0usize, 0usize); 2];
        for e in eye_ranges.iter_mut() {
            *e = (r.u32()? as usize, r.u32()? as usize);
        }
        let n_all = n_draw.checked_add(n_extra).ok_or("a count overflows")?;
        let base = r.vec3s(n_all)?;
        let base_normals = r.vec3s(n_draw)?;
        let raw = r.take(n_draw.checked_mul(32).ok_or("a count overflows")?)?;
        let attrs: Vec<[f32; 8]> = raw
            .chunks_exact(32)
            .map(|c| {
                let mut a = [0.0f32; 8];
                for (k, v) in a.iter_mut().enumerate() {
                    *v = f32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]]);
                }
                a
            })
            .collect();
        let edges = r.u32s(n_edge_idx)?;
        let tris = r.u32s(n_tri_idx)?;
        let sphere = r.vec3s(n_sphere)?;

        let mut named: HashMap<String, Vec<u32>> = HashMap::new();
        for _ in 0..n_landmarks {
            let name = r.name()?;
            let n = r.count(&name, 4)?;
            named.insert(name, r.u32s(n)?);
        }
        let mut targets = Vec::with_capacity(n_targets);
        let mut by_name = HashMap::new();
        for k in 0..n_targets {
            let name = r.name()?;
            let n = r.count(&name, 16)?;
            let idx = r.u32s(n)?;
            let delta = r.vec3s(n)?;
            if let Some(&bad) = idx.iter().find(|&&i| i as usize >= n_all) {
                return Err(format!("{name} moves vertex {bad}, past the {n_all} there are"));
            }
            by_name.insert(name.clone(), k);
            targets.push(Target { name, idx, delta });
        }
        if r.at != bytes.len() {
            return Err(format!("{} bytes left over after the last shape", bytes.len() - r.at));
        }

        // Every index is checked once here, so the arithmetic below can index
        // without a check that would fail only on a file that never loads.
        if edges.len() % 2 != 0 || tris.len() % 3 != 0 {
            return Err("an edge or triangle list that does not divide into whole edges or triangles".into());
        }
        if edges.iter().chain(tris.iter()).any(|&i| i as usize >= n_draw) {
            return Err("an edge or triangle refers to a vertex that is not drawn".into());
        }
        for (start, n) in eye_ranges {
            if n != n_sphere || start.checked_add(n).map_or(true, |end| end > n_draw) {
                return Err("an eyeball's range does not fit the drawn vertices".into());
            }
        }
        let mut get = |name: &str| -> Result<Vec<u32>, String> {
            let v = named.remove(name).ok_or(format!("no `{name}` landmark"))?;
            if v.is_empty() || v.iter().any(|&i| i as usize >= n_all) {
                return Err(format!("the `{name}` landmark is empty or out of range"));
            }
            Ok(v)
        };
        let landmarks = Landmarks {
            eye_centre: [get("eye-centre-l")?, get("eye-centre-r")?],
            eye_size: [get("eye-size-l")?, get("eye-size-r")?],
            ears: get("ears")?,
            anchor: get("head-anchor")?,
        };
        let mut body = Body {
            n_draw,
            base,
            base_normals,
            attrs,
            edges,
            tris,
            sphere,
            eye_ranges,
            ao_scale,
            hinge_offset,
            landmarks,
            targets,
            by_name,
            anchor0: [0.0; 3],
        };
        let base: Vec<[f64; 3]> = body.base.iter().map(|&p| wide(p)).collect();
        body.anchor0 = mean(&base, &body.landmarks.anchor);
        Ok(body)
    }

    /// The index of a target by its name in the file, e.g. `nose/nose-width1-incr`.
    pub fn target(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    pub fn target_name(&self, t: usize) -> &str {
        &self.targets[t].name
    }

    /// Per drawn vertex, the static weights: occlusion, jaw, lid, part, fade,
    /// chest, lift and a spare.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn attrs(&self, v: usize) -> [f32; 8] {
        self.attrs[v]
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn base_normal(&self, v: usize) -> [f32; 3] {
        self.base_normals[v]
    }

    /// The shape for these target weights: summed onto the base, the head held
    /// still, the eyes placed, and the normals and occlusion recomputed.
    pub fn pose(&self, weights: &[(usize, f32)]) -> Posed {
        let mut q: Vec<[f64; 3]> = self.base.iter().map(|&p| wide(p)).collect();
        for &(t, w) in weights {
            if w == 0.0 {
                continue;
            }
            let w = w as f64;
            let target = &self.targets[t];
            for (&i, d) in target.idx.iter().zip(&target.delta) {
                let p = &mut q[i as usize];
                p[0] += w * d[0] as f64;
                p[1] += w * d[1] as f64;
                p[2] += w * d[2] as f64;
            }
        }
        // Hold the head still. A shape that lengthens the neck or carries the
        // head forward moves the joint the head turns on; the window keeps the
        // face where the scene puts it and lets the body take up the change.
        let moved = sub(mean(&q, &self.landmarks.anchor), self.anchor0);
        for p in q.iter_mut() {
            *p = sub(*p, moved);
        }
        let rig = self.rig_of(&q);
        for (side, &(start, n)) in self.eye_ranges.iter().enumerate() {
            let c = [rig.eyes[side][0] as f64, rig.eyes[side][1] as f64, rig.eyes[side][2] as f64];
            let radius = rig.eyes[side][3] as f64;
            for k in 0..n {
                let s = wide(self.sphere[k]);
                q[start + k] = [c[0] + radius * s[0], c[1] + radius * s[1], c[2] + radius * s[2]];
            }
        }
        let drawn = &q[..self.n_draw];
        let normals = normals(drawn, &self.tris);
        let ao = cavity(drawn, &normals, &self.edges)
            .into_iter()
            .map(|c| (c / self.ao_scale).clamp(-1.0, 1.0) as f32)
            .collect();
        Posed {
            positions: drawn.iter().map(|&p| narrow(p)).collect(),
            normals: normals.into_iter().map(narrow).collect(),
            ao,
            rig,
        }
    }

    fn rig_of(&self, q: &[[f64; 3]]) -> Rig {
        let mut eyes = [[0.0f32; 4]; 2];
        for side in 0..2 {
            let c = mean(q, &self.landmarks.eye_centre[side]);
            let (lo, hi) = self.landmarks.eye_size[side]
                .iter()
                .map(|&i| q[i as usize][1])
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), y| (lo.min(y), hi.max(y)));
            eyes[side] = [c[0] as f32, c[1] as f32, c[2] as f32, (0.5 * (hi - lo)) as f32];
        }
        let ear = mean(q, &self.landmarks.ears);
        Rig {
            eyes,
            hinge: [
                (ear[1] + self.hinge_offset[0]) as f32,
                (ear[2] + self.hinge_offset[1]) as f32,
            ],
        }
    }

    /// How much each drawn vertex moves under these targets at full weight,
    /// scaled so the most-moved vertex is 1. What the creator lights up while
    /// a control is hovered, so a person can see what a slider will touch.
    pub fn influence(&self, targets: &[usize]) -> Vec<f32> {
        let mut m = vec![0.0f32; self.n_draw];
        for &t in targets {
            let target = &self.targets[t];
            for (&i, d) in target.idx.iter().zip(&target.delta) {
                if (i as usize) < self.n_draw {
                    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    m[i as usize] = m[i as usize].max(len);
                }
            }
        }
        let top = m.iter().copied().fold(0.0f32, f32::max);
        if top > 0.0 {
            for v in m.iter_mut() {
                *v /= top;
            }
        }
        m
    }

    /// The interleaved vertex data the GPU draws, `FLOATS_PER_VERTEX` per vertex.
    pub fn vertex_data(&self, posed: &Posed, highlight: Option<&[f32]>) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_draw * FLOATS_PER_VERTEX);
        for v in 0..self.n_draw {
            let a = self.attrs[v];
            let p = posed.positions[v];
            let n = posed.normals[v];
            out.extend_from_slice(&[p[0], p[1], p[2], n[0], n[1], n[2]]);
            out.extend_from_slice(&[posed.ao[v], a[1], a[2], a[3]]);
            out.extend_from_slice(&[a[4], a[5], a[6], highlight.map_or(0.0, |h| h[v])]);
        }
        out
    }
}

fn mean(q: &[[f64; 3]], idx: &[u32]) -> [f64; 3] {
    let s = idx.iter().fold([0.0; 3], |s, &i| add(s, q[i as usize]));
    let n = idx.len() as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

/// Area-weighted vertex normals from the triangles, exactly as the bake does.
fn normals(q: &[[f64; 3]], tris: &[u32]) -> Vec<[f64; 3]> {
    let mut n = vec![[0.0f64; 3]; q.len()];
    for t in tris.chunks_exact(3) {
        let (a, b, c) = (q[t[0] as usize], q[t[1] as usize], q[t[2] as usize]);
        let f = cross(sub(b, a), sub(c, a));
        for &i in t {
            n[i as usize] = add(n[i as usize], f);
        }
    }
    for v in n.iter_mut() {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let len = if len > 1e-12 { len } else { 1.0 };
        *v = [v[0] / len, v[1] / len, v[2] / len];
    }
    n
}

/// How far each vertex sits inside the average of its neighbours, along its normal.
fn cavity(q: &[[f64; 3]], n: &[[f64; 3]], edges: &[u32]) -> Vec<f64> {
    let mut s = vec![[0.0f64; 3]; q.len()];
    let mut count = vec![0.0f64; q.len()];
    for e in edges.chunks_exact(2) {
        let (a, b) = (e[0] as usize, e[1] as usize);
        s[a] = add(s[a], q[b]);
        s[b] = add(s[b], q[a]);
        count[a] += 1.0;
        count[b] += 1.0;
    }
    (0..q.len())
        .map(|v| {
            let c = count[v].max(1.0);
            let avg = [s[v][0] / c, s[v][1] / c, s[v][2] / c];
            let d = sub(avg, q[v]);
            d[0] * n[v][0] + d[1] * n[v][1] + d[2] * n[v][2]
        })
        .collect()
}

#[cfg(test)]
#[path = "body_tests.rs"]
mod tests;
