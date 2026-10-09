// Sinai: a real bust, moving the way a body moves.
//
// Everything this file used to contain was an attempt to derive a human head
// from arithmetic -- twenty-four gaussians on a radius, then a face plane, then
// a mandible path, then vector displacements for the forms that project. Each
// round was better than the last and none was a face, because a head is a shape
// somebody measured and that was a shape I was guessing.
//
// The geometry is the MakeHuman base mesh, released CC0 in September 2020,
// cropped to a bust by bake_body.py and shaped on the CPU by body.rs: whatever a
// person set in the creator, and whatever expression Sinai is wearing, arrive
// here as positions and normals already. What is left here is what a shader
// should do: pose it, light it, and let it dissolve at its edges.
//
// The first version to use that mesh still PAINTED the eyes on: an ellipse of
// shading where I thought an eye went. It went somewhere else, and read as two
// dark rings floating on its temples. Nothing is painted on now. The eyes are
// eyeballs, the blink rotates the real eyelid about the real eye centre, and
// the mouth opens by swinging the mandible about its hinge -- which is why the
// lower lip, the chin and the lower teeth all move together and the upper lip
// stays where it is, without any of that being arranged.

struct Head {
    mvp    : mat4x4<f32>,
    mv     : mat4x4<f32>,
    params : vec4<f32>,      // mouth open, blink, time, breath (-1 out .. 1 in)
    tint   : vec4<f32>,      // the lattice's colour, and its strength
    light  : vec4<f32>,      // key direction, fill weight
    eye_l  : vec4<f32>,      // Sinai's left eye: centre, radius
    eye_r  : vec4<f32>,      // its right
    rig    : vec4<f32>,      // jaw hinge y, jaw hinge z, highlight strength, _
    fill   : vec4<f32>,      // the body's colour where it is lit
    shadow : vec4<f32>,      // and where it is not
    glow   : vec4<f32>,      // the colour the bust dissolves into at its cuts; w: how strongly
    iris   : vec4<f32>,      // the irises' colour; w: how much of it shows
};

@group(0) @binding(0) var<uniform> head : Head;

// How far the jaw drops at a full vowel, and how far the lid sweeps to close.
// Both are angles because both are hinges.
const JAW_OPEN : f32 = 0.31;
const LID_SHUT : f32 = 1.05;

// A breath, in head units: the ribcage swells along its surface and the
// shoulder girdle rises. Small, because quiet breathing is small; a chest that
// heaves reads as distress.
const BREATH_SWELL : f32 = 0.022;
const BREATH_LIFT  : f32 = 0.018;

const SKIN : f32 = 0.0;
const BALL : f32 = 1.0;
const TEETH_U : f32 = 2.0;

/// Rotate a point and a direction together about an axis parallel to x.
fn hinge(p: vec3<f32>, n: vec3<f32>, at: vec2<f32>, a: f32) -> array<vec3<f32>, 2> {
    let c = cos(a);
    let s = sin(a);
    let d = vec2<f32>(p.y - at.x, p.z - at.y);
    return array<vec3<f32>, 2>(
        vec3<f32>(p.x, at.x + c * d.x - s * d.y, at.y + s * d.x + c * d.y),
        vec3<f32>(n.x, c * n.y - s * n.z, s * n.y + c * n.z),
    );
}

/// The eye a vertex belongs to: Sinai's left is +x. Once a person has shaped
/// the face the two are not mirror images, so neither is derived from the other.
fn eye_for(x: f32) -> vec4<f32> {
    return select(head.eye_r, head.eye_l, x > 0.0);
}

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) shade     : f32,
    @location(1) facing    : f32,
    /// Where this fragment sits on the eyeball, as a direction from its centre.
    /// The eye is shaded per FRAGMENT rather than per vertex: the ball is
    /// a few hundred vertices, so an iris computed at the vertices and
    /// interpolated came out as a dark rectangle with corners.
    @location(2) ball      : vec3<f32>,
    @location(3) @interpolate(flat) part : f32,
    /// 0 on a cut through the bust, 1 a band's width away from any cut.
    @location(4) fade      : f32,
    @location(5) highlight : f32,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>,
           @location(1) nrm: vec3<f32>,
           @location(2) w:   vec4<f32>,      // occlusion, jaw, lid, part
           @location(3) w2:  vec4<f32>) -> VsOut {   // fade, chest, lift, highlight
    let ao = w.x;
    let part = w.w;
    var p = pos;
    var n = nrm;

    // The breath. Below the neck only, so it never reaches the face.
    let breath = head.params.w;
    p = p + n * (breath * BREATH_SWELL * w2.y) + vec3<f32>(0.0, breath * BREATH_LIFT * w2.z, 0.0);

    // The mandible. One hinge, one angle, and the weight says how much of the
    // jaw each vertex is -- so the chin swings, the cheek follows it part way,
    // and the ear does not move at all.
    if (w.y > 0.001) {
        let r = hinge(p, n, head.rig.xy, head.params.x * JAW_OPEN * w.y);
        p = r[0];
        n = r[1];
    }

    // The lid, about the eye it covers. A blink CLOSES: an earlier version
    // multiplied in a wider dark gaussian as the blink rose, which grew the
    // dark patch instead, and what showed was that its eyes got wider
    // when it blinked. They did. A lid that is geometry cannot do that.
    if (w.z > 0.001) {
        let e = eye_for(pos.x);
        let r = hinge(p, n, e.yz, head.params.y * LID_SHUT * w.z);
        p = r[0];
        n = r[1];
    }

    // Light. Key from the side, because a light from straight ahead says
    // nothing about a face pointing at you; fill from below so the nose and
    // lips do not sit in their own shadow.
    let key = normalize(head.light.xyz);
    let fill = normalize(vec3<f32>(-0.25, -0.65, 0.72));
    let lit = max(0.0, dot(n, key)) * 0.78 + max(0.0, dot(n, fill)) * head.light.w;
    let rim = 1.0 - abs(n.z);

    var shade = 0.13 + 0.20 * rim + 0.64 * lit;

    // Occlusion, recomputed whenever the shape changes, over the real topology.
    // Hollows are what a face reads by: the sockets, the nostrils, the seam
    // between the lips and the crease under the brow all face the same way as
    // the skin beside them and are invisible to a light alone.
    shade = shade * (1.0 - 0.72 * max(0.0, ao)) + 0.20 * max(0.0, -ao);

    var ball = vec3<f32>(0.0, 0.0, 1.0);
    if (part == BALL) {
        ball = normalize(pos - eye_for(pos.x).xyz);
    } else if (part >= TEETH_U) {
        shade = 0.72 + 0.34 * lit;
    }

    let vpos = head.mv * vec4<f32>(p, 1.0);
    let vn = (head.mv * vec4<f32>(n, 0.0)).xyz;

    var out : VsOut;
    out.pos = head.mvp * vec4<f32>(p, 1.0);
    out.shade = shade;
    out.facing = dot(normalize(vn), normalize(-vpos.xyz));
    out.ball = ball;
    out.part = part;
    out.fade = w2.x;
    out.highlight = w2.w;
    return out;
}

/// Sclera, iris, pupil and a catchlight, each a cone about the direction it is
/// looking and sized off a real eye: the iris subtends about thirty degrees of
/// the ball and the pupil about ten. Returns the brightness, and how much of
/// the iris this point is, so the iris can take a colour of its own.
fn eye_shade(dir: vec3<f32>) -> vec2<f32> {
    let g = normalize(dir);
    let iris = smoothstep(0.800, 0.870, g.z);
    let pupil = smoothstep(0.965, 0.985, g.z);
    let spark = smoothstep(0.988, 0.9995,
        dot(g, normalize(vec3<f32>(-0.34, 0.44, 0.83))));
    // Bright enough to read as an eye from across the room. At 1.05 the sclera
    // was the same value as the skin around it and both eyes looked like empty
    // sockets in the window as it actually runs. Wide enough an iris that the
    // white does not dominate the opening, which reads as a stare.
    return vec2<f32>(1.30 - 0.86 * iris - 0.30 * pupil + 0.90 * spark, iris * (1.0 - pupil) * (1.0 - spark));
}

/// How strongly the creator lights up this fragment: the whole region a hovered
/// control moves shows, not only its peak, so a person can see what it touches.
fn lit_up(highlight: f32) -> f32 {
    let h = clamp(highlight * head.rig.z, 0.0, 1.0);
    return select(0.0, 0.35 + 0.65 * h, h > 0.002);
}

/// A fixed per-pixel threshold in 0..1, so a dissolve thins the surface out
/// pixel by pixel instead of turning the whole of it translucent.
fn dither(at: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(floor(at), vec2<f32>(0.06711056, 0.00583715))));
}

// The lattice.
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    if (in.facing <= 0.0) { discard; }
    var shade = in.shade;
    var colour = head.tint.rgb;
    if (in.part == BALL) {
        let e = eye_shade(in.ball);
        shade = e.x;
        colour = mix(colour, head.iris.rgb, e.y * head.iris.w);
    }
    // Towards a cut the lattice burns into the glow and thins away, so the
    // bust ends the way a projection would rather than on a sawn edge.
    let edge = 1.0 - smoothstep(0.0, 1.0, in.fade);
    colour = mix(colour, head.glow.rgb, edge * head.glow.w);
    let shown = lit_up(in.highlight);
    colour = mix(colour, vec3<f32>(1.0, 0.97, 0.88), shown * 0.6);
    let a = clamp(shade, 0.0, 1.6) * head.tint.a * (1.0 + 1.6 * shown) * smoothstep(0.0, 0.55, in.fade);
    return vec4<f32>(colour * a, a);
}

// The being's body: opaque and nearly black, under the lattice so the valley stops
// showing through it. Shaded rather than flat, because a pure silhouette reads
// as a hole cut in the picture.
@fragment
fn fs_fill(in: VsOut) -> @location(0) vec4<f32> {
    // Towards a cut the body dissolves pixel by pixel, so the lattice on its
    // far side shows through where a projection would thin out.
    if (in.fade < dither(in.pos.xy) * 0.999) { discard; }
    let lo = head.shadow.rgb;
    // A surface facing away is the INSIDE of the being -- the back of the throat when
    // its jaw is down, the inside of a cheek. Discarding it left a hole with the
    // valley shining through, so it is drawn instead, near black: a mouth that
    // opens onto darkness rather than onto the landscape behind the being.
    if (in.facing <= 0.0) {
        return vec4<f32>(lo * 0.55, 1.0);
    }
    var shade = in.shade;
    var iris = 0.0;
    if (in.part == BALL) {
        let e = eye_shade(in.ball);
        shade = e.x;
        iris = e.y * head.iris.w;
    }
    let k = clamp(shade, 0.0, 1.0);
    var colour = mix(lo, head.fill.rgb, k);
    colour = mix(colour, head.iris.rgb * (0.25 + 0.35 * k), iris * 0.6);
    colour = colour + head.tint.rgb * 0.09 * lit_up(in.highlight);
    return vec4<f32>(colour, 1.0);
}
