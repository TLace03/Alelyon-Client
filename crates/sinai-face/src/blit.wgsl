// The scene, composited back over the window.
//
// Angel and its valley are drawn into a texture of their own with a depth
// buffer, because the pass egui hands a paint callback has no depth attachment
// and cannot be given one. Without depth, every line of its far side drew over
// its near side: the facing test removes the half of the surface that points
// away, but nothing was left to say that its occiput is behind its face. Giving
// it eyeballs and teeth made that unmistakable -- the teeth drew straight
// through its lips and sat on its chin like a grille.
//
// So: one offscreen pass with depth, then this.

@group(0) @binding(0) var scene : texture_2d<f32>;
@group(0) @binding(1) var samp  : sampler;

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv        : vec2<f32>,
};

// One triangle large enough to cover the viewport, which beats two for a full
// screen pass: no seam down the diagonal and half the vertices.
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out : VsOut;
    out.pos = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Already premultiplied: everything that drew into the offscreen target
    // wrote colour times alpha, which is what the blend below expects.
    return textureSample(scene, samp, in.uv);
}
