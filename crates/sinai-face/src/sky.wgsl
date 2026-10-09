// The sky behind the valley, at the hour it actually is.
//
// The browser version drew this and the native window lost it, so Sinai has been
// travelling through a valley under a flat black ceiling since the port.
//
// The sun is where the sun is. Declination from the day of the year, hour angle
// from the local clock, elevation from those two and a latitude -- the one term
// this machine does not already know, which is why it is a setting rather than
// a lookup. Asking the network where somebody lives in order to render a
// gradient is not a trade worth making quietly.
//
// Day stays muted on purpose. A literal blue noon would wash a gold wireframe
// off Sinai's own screen, so the palette runs from near-black at night to a dim
// warm grey at midday, and the drama lives in the two twilights.

struct Sky {
    top    : vec4<f32>,      // rgb, and the horizon's height in clip space
    mid    : vec4<f32>,      // rgb, and how much of the night is left (0..1)
    hor    : vec4<f32>,      // rgb, and the sun's elevation, -1..1
    sun    : vec4<f32>,      // x in clip space, y in clip space, time, cloud
};

@group(0) @binding(0) var<uniform> sky : Sky;

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv        : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out : VsOut;
    out.pos = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);   // clip space
    return out;
}

// ---- the stars' hash --------------------------------------------------------
// The star grid is hashed with integers, so every adapter draws the same stars.
//
// The hash this replaces was fract(sin(dot(p, (127.1, 311.7))) * 43758.5453),
// the sine hash the valley also used. WGSL bounds sin's error only between -pi
// and pi; here its argument runs to about 24,000, and the factor of 43758 turns
// whatever error an adapter makes there into a different value. So the stars
// at the top of the sky differed between the RX 9070 XT and the software
// rasteriser (2026-10-02).
//
// This is terrain.wgsl's hash, repeated because WGSL has no includes:
// xxHash32 (Yann Collet's) of the cell's two coordinates as eight little-endian
// bytes, seeded with `salt`, in u32 arithmetic that wraps the same way on every
// adapter. sky_tests.rs holds this copy to xxHash's own value, as
// terrain_tests.rs holds the valley's.
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

// A star's three values come from three streams. The old hash read them at
// g, g + 3.7 and g + 9.1; the valley's streams are 0 and 1, so the stars take
// the next three and no two values in the scene share one.
const STAR_SALT : u32 = 2u;     // whether the cell has a star, how bright it is, its twinkle
const STAR_X_SALT : u32 = 3u;   // where in its cell the star sits, across
const STAR_Y_SALT : u32 = 4u;   // and up

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let horizon = sky.top.w;
    let night = sky.mid.w;

    // Height above the horizon, 0 at it and 1 at the top of the window.
    let h = clamp((in.uv.y - horizon) / max(1.0 - horizon, 1e-3), 0.0, 1.0);

    // Two stops rather than three interpolations: horizon to middle over the
    // first third, middle to top over the rest, which is roughly how a real
    // gradient sits and costs one mix each.
    var c = mix(sky.hor.rgb, sky.mid.rgb, smoothstep(0.0, 0.34, h));
    c = mix(c, sky.top.rgb, smoothstep(0.30, 1.0, h));

    // Stars, in the half of the sky above the horizon, and only as far as the
    // night allows. They sit on a grid so that they do not crawl when the
    // window is resized: the cell is a fraction of clip space, not of pixels.
    if (night > 0.01 && in.uv.y > horizon) {
        let cell = 0.022;
        let g = floor(in.uv / cell);
        let k = vec2<i32>(g);
        let r = hash2(k, STAR_SALT);
        if (r > 0.986) {
            let centre = (g + vec2<f32>(hash2(k, STAR_X_SALT), hash2(k, STAR_Y_SALT))) * cell;
            let d = length((in.uv - centre) / vec2<f32>(cell * 0.30, cell * 0.30));
            // Brighter ones are rarer, and every one twinkles at its own rate.
            let mag = 0.35 + 0.65 * fract(r * 71.3);
            let tw = 0.75 + 0.25 * sin(sky.sun.z * (0.7 + r) + r * 40.0);
            c = c + vec3<f32>(0.85, 0.87, 1.0) * exp(-d * d * 2.0) * mag * tw * night * h;
        }
    }

    // The sun, or the moon after it has set. A disc with a wide, weak halo:
    // the halo is what makes a gradient read as having something in it.
    let aspect = vec2<f32>(1.0, 0.62);           // the window is taller than wide
    let d = length((in.uv - sky.sun.xy) * aspect);
    let above = step(horizon - 0.06, sky.sun.y);
    let elev = sky.hor.w;
    let warm = clamp((elev + 0.25) / 0.5, 0.0, 1.0);
    let body = mix(vec3<f32>(0.78, 0.80, 0.92), vec3<f32>(1.0, 0.86, 0.60), warm);
    c = c + body * (exp(-d * d * 900.0) * 0.85 + exp(-d * d * 22.0) * 0.16) * above;

    // Cloud thins everything: a flat wash toward the horizon colour, which is
    // what an overcast sky is.
    c = mix(c, sky.hor.rgb * 1.15, sky.sun.w * 0.55 * (0.35 + 0.65 * (1.0 - h)));

    return vec4<f32>(c, 1.0);
}
