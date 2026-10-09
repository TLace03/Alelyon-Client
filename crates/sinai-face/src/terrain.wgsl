// The valley, computed entirely on the GPU.
//
// The browser version generated every vertex on the CPU each frame and cached
// noise in a ring buffer to make that affordable. None of that is needed here:
// a vertex shader runs this for a quarter of a million vertices in less time
// than the JavaScript took to walk forty rows, so the mesh is denser AND the
// per-frame CPU cost is a single uniform write.
//
// The grid itself is uploaded once and never changes. What moves is `travel`:
// rows hold station in view space while the NOISE is sampled at an absolute
// world position, which is what makes the terrain endless rather than looping.

struct Camera {
    view_proj : mat4x4<f32>,
    // x: travel, y: theme_mix, z: theme_a, w: theme_b
    motion    : vec4<f32>,
    // x: road_half, y: dz, z: col_dx, w: daylight
    params    : vec4<f32>,
    // rgb line colour, a: master alpha
    tint      : vec4<f32>,
    // x: fade_start, y: fade_end, z: cam_y, w: unused
    fade      : vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam : Camera;

// ---- value noise ----------------------------------------------------------
// The lattice is hashed with integers, so every adapter draws the same valley.
//
// The browser's hash was fract(sin(dot(p, (127.1, 311.7))) * 43758.5453123).
// WGSL bounds sin's error only between -pi and pi; here its argument is in the
// tens of thousands and grows as the valley scrolls, and the factor of 43758
// turns whatever error an adapter makes there into a different answer. So the
// RX 9070 XT and the software rasteriser drew different ridgelines from the
// same uniforms (2026-10-02). u32 multiplies, shifts and xors wrap the same way
// on every adapter.
//
// The hash is xxHash32 (Yann Collet's) of the cell's two coordinates as eight
// little-endian bytes, seeded with `salt`: a published function, so the test
// holds the shader to xxHash's own vectors rather than to a copy of itself.
const PRIME32_2 : u32 = 0x85EBCA77u;
const PRIME32_3 : u32 = 0xC2B2AE3Du;
const PRIME32_4 : u32 = 0x27D4EB2Fu;
const PRIME32_5 : u32 = 0x165667B1u;

fn xxh32_cell(cell: vec2<i32>, salt: u32) -> u32 {
    let w = bitcast<vec2<u32>>(cell);
    var h = salt + PRIME32_5 + 8u;                  // seed, prime 5, length
    h = h + w.x * PRIME32_3;
    h = ((h << 17u) | (h >> 15u)) * PRIME32_4;
    h = h + w.y * PRIME32_3;
    h = ((h << 17u) | (h >> 15u)) * PRIME32_4;
    h = (h ^ (h >> 15u)) * PRIME32_2;               // the avalanche
    h = (h ^ (h >> 13u)) * PRIME32_3;
    return h ^ (h >> 16u);
}

// The cell's value in [0, 1). Its top 24 bits convert to an f32 exactly and
// the scale is a power of two, so the float is the same bits everywhere too.
fn hash2(cell: vec2<i32>, salt: u32) -> f32 {
    return f32(xxh32_cell(cell, salt) >> 8u) * (1.0 / 16777216.0);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let k = vec2<i32>(i);
    let a = hash2(k, 0u);
    let b = hash2(k + vec2<i32>(1, 0), 0u);
    let c = hash2(k + vec2<i32>(0, 1), 0u);
    let d = hash2(k + vec2<i32>(1, 1), 0u);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>, oct: i32) -> f32 {
    var amp = 0.5;
    var q = 1.0;
    var v = 0.0;
    for (var i = 0; i < oct; i = i + 1) {
        v = v + amp * vnoise(p * q);
        q = q * 2.07;
        amp = amp * 0.5;
    }
    return v;
}

fn ridged(p: vec2<f32>, oct: i32) -> f32 {
    var amp = 0.5;
    var q = 1.0;
    var v = 0.0;
    for (var i = 0; i < oct; i = i + 1) {
        v = v + amp * (1.0 - abs(2.0 * vnoise(p * q) - 1.0));
        q = q * 2.11;
        amp = amp * 0.5;
    }
    return v;
}

// ---- the four places ------------------------------------------------------
// Same shapings as the browser scene, so this is a port rather than a redesign.
fn shaped(kind: f32, n: vec2<f32>) -> f32 {
    let r = ridged(n, 4);
    let f = fbm(n * 0.7, 3);
    // The blocks draw from their own stream (salt 1), so a block's height is
    // not the noise lattice's value at the same cell.
    let blk = hash2(vec2<i32>(floor(vec2<f32>(n.x * 4.35, n.y * 1.21))), 1u);
    if (kind < 0.5) {
        return pow(r, 1.40) * 4.6;            // alpine: peaks
    } else if (kind < 1.5) {
        return floor(f * 5.0) / 5.0 * 3.8 + 0.35;   // western: mesas
    } else if (kind < 2.5) {
        return 0.5 + blk * blk * 6.4;         // urban: blocks
    }
    return pow(r, 2.9) * 8.2;                 // alien: spires
}

// Flat road down the middle, ground rising either side, bigger country further
// out. The smoothstep is what keeps the shoulder from being a cliff.
fn ground_height(world_x: f32, world_z: f32) -> f32 {
    let n = vec2<f32>(world_x * 0.148, world_z * 0.0682);
    var h = shaped(cam.motion.z, n);
    if (cam.motion.y > 0.0) {
        h = h + (shaped(cam.motion.w, n) - h) * cam.motion.y;
    }
    let ax = abs(world_x);
    let road = cam.params.x;
    if (ax <= road) { return 0.0; }
    var m = clamp((ax - road) / (road * 0.9), 0.0, 1.0);
    m = m * m * (3.0 - 2.0 * m);
    return h * m * (0.70 + 1.35 * min(1.0, ax / 24.0));
}

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) depth     : f32,
};

// The vertex buffer holds grid COORDINATES, not positions: column index and
// row index. Everything else is derived here, so the buffer is uploaded once
// and never touched again.
@vertex
fn vs_main(@location(0) grid: vec2<f32>) -> VsOut {
    let travel = cam.motion.x;
    let dz = cam.params.y;
    let dx = cam.params.z;

    // Columns bunch near the road and spread at the sides, where perspective
    // crushes them together anyway. NORMALISED first: the browser version's
    // quadratic term was tuned for twenty columns, and feeding it a raw index
    // of 128 put the outermost column four hundred world units out instead of
    // sixty-five, which is why the first render had wings.
    let c = grid.x;
    let nx = cam.params.w;
    let u = c / nx;
    let au = abs(u);
    let world_x = sign(c) * (au * 15.0 + au * au * 25.0);

    // Rows hold station in view space; the noise is sampled at the absolute
    // world row, so the ground is endless instead of a loop.
    let row_base = floor(travel / dz);
    let z_view = grid.y * dz - (travel - row_base * dz);
    let world_z = (row_base + grid.y) * dz;

    let h = ground_height(world_x, world_z);

    var out : VsOut;
    out.pos = cam.view_proj * vec4<f32>(world_x, h, z_view, 1.0);
    out.depth = z_view;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Distance fade, so the mesh dissolves into the sky rather than stopping.
    let t = clamp((in.depth - cam.fade.x) / max(1.0, cam.fade.y - cam.fade.x), 0.0, 1.0);
    let a = cam.tint.a * (1.0 - t) * (1.0 - t);
    return vec4<f32>(cam.tint.rgb * a, a);
}
