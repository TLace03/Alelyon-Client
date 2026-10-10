//! IEEE 754 binary16 conversions on the host, for the depth channel.
//!
//! The device converts with GLSL `packHalf2x16`, whose rounding the GLSL
//! specification leaves to the implementation; the host reference rounds to
//! nearest, ties to even. Tests therefore compare depth in metres within a
//! tolerance of one binary16 step, never bit for bit across host and device.

/// The binary16 bits of positive infinity: the depth of a pixel that hit
/// nothing.
pub const F16_INFINITY: u16 = 0x7C00;

/// `x` as binary16 bits, rounded to nearest with ties to even. Overflow gives
/// infinity; NaN gives a quiet NaN.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x007F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x0200 } else { 0 };
    }
    // unbiased exponent
    let e = exp - 127;
    if e > 15 {
        return sign | 0x7C00;
    }
    if e >= -14 {
        // normal: 10 mantissa bits, round the 13 dropped ones
        let half_exp = (e + 15) as u32;
        let mut h = (half_exp << 10) | (man >> 13);
        let rest = man & 0x1FFF;
        if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
            h += 1; // a carry into the exponent is the correct result, up to infinity
        }
        return sign | h as u16;
    }
    if e < -25 {
        return sign;
    }
    // subnormal: value = m * 2^-24 with the implicit bit restored
    let full = man | 0x0080_0000;
    let shift = (-e - 14 + 13) as u32; // 14..=24
    let mut h = full >> shift;
    let rest = full & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    if rest > halfway || (rest == halfway && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// The value of binary16 bits.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = (h >> 10) & 0x1F;
    let man = (h & 0x03FF) as f32;
    match exp {
        0 => sign * man * (2.0f32).powi(-24),
        0x1F => {
            if man == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + man / 1024.0) * (2.0f32).powi(exp as i32 - 15),
    }
}

/// The spacing of binary16 values at `x` (one unit in the last place).
pub fn f16_ulp(x: f32) -> f32 {
    let a = x.abs();
    if a < (2.0f32).powi(-14) {
        return (2.0f32).powi(-24);
    }
    let e = a.log2().floor() as i32;
    (2.0f32).powi(e - 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values_round_trip() {
        for (x, bits) in [
            (0.0f32, 0x0000u16),
            (1.0, 0x3C00),
            (-2.0, 0xC000),
            (0.5, 0x3800),
            (65504.0, 0x7BFF),
            (2.0f32.powi(-14), 0x0400),
            (2.0f32.powi(-24), 0x0001),
        ] {
            assert_eq!(f32_to_f16(x), bits, "{x}");
            assert_eq!(f16_to_f32(bits), x);
        }
    }

    #[test]
    fn rounding_is_to_nearest_even_and_overflow_is_infinity() {
        // 1 + 2^-11 is halfway between 1 and 1 + 2^-10: ties go to the even 1
        assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3C00);
        // 1 + 3 * 2^-11 is halfway between 1 + 2^-10 and 1 + 2^-9: even is the upper
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3C02);
        assert_eq!(f32_to_f16(70000.0), F16_INFINITY);
        assert_eq!(f32_to_f16(f32::INFINITY), F16_INFINITY);
        assert!(f16_to_f32(F16_INFINITY).is_infinite());
    }

    #[test]
    fn every_finite_half_converts_back_to_itself() {
        for h in 0u16..0x7C00 {
            assert_eq!(f32_to_f16(f16_to_f32(h)), h, "{h:#06x}");
        }
    }
}
