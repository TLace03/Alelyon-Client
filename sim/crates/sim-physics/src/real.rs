//! The number type the whole engine is generic over: `f64` or `f32`.
//!
//! `f64` is what the engine is held to MuJoCo with; `f32` is what the GPU port
//! will run, and it is measured against the same golden values here.
//!
//! Invariants:
//! - [`Real`] has exactly the operations the ported MuJoCo code uses: the four
//!   arithmetic operators, negation, comparison, `sqrt`, `sin`, `cos`, `atan2`,
//!   `abs`, `powf` (the constraint impedance of a power other than 1 and 2), and
//!   conversion to and from `f64`. There is no fused multiply-add in
//!   it and the engine never calls one: Rust does not fuse `a * b + c` on its own
//!   (it has no `-ffast-math` and no contraction), and this crate keeps it that
//!   way so that the GPU port, which must not fuse either, matches the operation
//!   order written here.
//! - `from_f64` rounds to nearest (an `as` cast); `to_f64` is exact.
//! - `sin`, `cos`, `atan2` and `sqrt` are the platform library's (`std`); `sqrt`
//!   is correctly rounded everywhere, the others are not claimed to be bit
//!   identical across platforms or to a GPU's.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A floating-point number type the engine can run in.
pub trait Real:
    Copy
    + Debug
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + 'static
{
    /// Zero.
    const ZERO: Self;
    /// One.
    const ONE: Self;
    /// MuJoCo's `mjBOXBOX_SEPEPS` (`engine_collision_box.c`): the rounding slack of the
    /// box-box separation tests, times the sum of the half-sizes. `1e-13` in `f64`, and
    /// the `mjUSESINGLE` value `1e-6` in `f32`.
    const BOXBOX_SEPEPS: Self;
    /// `mjBOXBOX_PAREPS`: the `sin^2` below which an edge-cross axis is noise. `1e-16`
    /// in `f64`, `1e-7` in `f32`.
    const BOXBOX_PAREPS: Self;
    /// `mjBOXBOX_SGNEPS`: the axis component below which a support corner is ambiguous.
    /// `1e-9` in `f64`, `1e-5` in `f32`.
    const BOXBOX_SGNEPS: Self;
    /// `mjBOXBOX_DUPEPS`: the squared relative radius for clip-vertex deduplication.
    /// `1e-14` in `f64`, `1e-10` in `f32`.
    const BOXBOX_DUPEPS: Self;
    /// `mjBOXBOX_EDGEBIAS`: the relative penalty on an edge axis against the best face
    /// axis (`1e-6` in both precisions).
    const BOXBOX_EDGEBIAS: Self;
    /// The squared length below which the cancelled axis residual of `mjc_PlaneCylinder`
    /// (`axis * (axis . normal) - normal`) is noise, not geometry: the "disk parallel to the
    /// plane" branch is taken for it. MuJoCo's test is `len_sqr >= mjMINVAL^2` (`1e-30`), which
    /// `f64` keeps bit for bit. In `f32` an upright cylinder's residual is the rounding of its
    /// rotation matrix (`1 - 6e-8` on the diagonal), about `1e-7` long, which is far above
    /// `1e-30`, so MuJoCo's test sends it down the general branch with a direction made of
    /// rounding. The `f32` value is the machine epsilon: a residual shorter than `sqrt(eps)`
    /// (`3.4e-4`, a tilt of 0.02 degrees) has a direction known to less than a few percent.
    const AXIS_RESIDUAL_SQR: Self;
    /// The relative size of a determinant `|det|` below which two segment axes count as
    /// parallel, in units of the product of their squared lengths (`ma * mc`): the test of
    /// `mjraw_CapsuleCapsule` and the edge test of `mjraw_CapsuleBox` is `|det| < mjMINVAL`
    /// (absolute, `1e-15`), to which this adds `PARALLEL_DET_REL * ma * mc`. `0` in `f64` (the
    /// test is MuJoCo's, bit for bit); in `f32`, `1e-6`: the determinant of exactly parallel
    /// axes is a rounding residue of a few units of `2^-24` times `ma * mc`, which is far above
    /// the absolute `1e-15` for axes of any usual length.
    const PARALLEL_DET_REL: Self;
    /// The relative slack of `mjraw_CapsuleBox`'s tie rule (`dist2 < bestdist - mjMINVAL`, "it
    /// fixes a numerical problem when the axis is numerically parallel to the box"): the
    /// candidate must beat the best squared distance by `mjMINVAL + TIE_REL * bestdist`. `0` in
    /// `f64`; `1e-6` in `f32`, where equal distances differ by rounding of that size.
    const TIE_REL: Self;
    /// MuJoCo's `mjMAXVAL`, the largest value of any state or distance (`1e10`).
    const MAXVAL: Self;
    /// Rounds an `f64` to this type (to nearest).
    fn from_f64(v: f64) -> Self;
    /// Widens to `f64` (exact).
    fn to_f64(self) -> f64;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Four-quadrant arctangent of `self / x`, `self` being the ordinate.
    fn atan2(self, x: Self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// `self` to the power `y` (MuJoCo's `mju_pow`, only for the constraint
    /// impedance with a power other than 1 and 2).
    fn powf(self, y: Self) -> Self;
}

macro_rules! impl_real {
    ($t:ty, $sep:expr, $par:expr, $sgn:expr, $dup:expr, $axres:expr, $pardet:expr, $tie:expr) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const BOXBOX_SEPEPS: Self = $sep;
            const BOXBOX_PAREPS: Self = $par;
            const BOXBOX_SGNEPS: Self = $sgn;
            const BOXBOX_DUPEPS: Self = $dup;
            const BOXBOX_EDGEBIAS: Self = 1e-6;
            const AXIS_RESIDUAL_SQR: Self = $axres;
            const PARALLEL_DET_REL: Self = $pardet;
            const TIE_REL: Self = $tie;
            const MAXVAL: Self = 1e10;
            #[inline]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                f64::from(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }
            #[inline]
            fn atan2(self, x: Self) -> Self {
                <$t>::atan2(self, x)
            }
            #[inline]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            #[inline]
            fn powf(self, y: Self) -> Self {
                <$t>::powf(self, y)
            }
        }
    };
}

// the box-box epsilons: MuJoCo's double-precision values, and its `mjUSESINGLE` values
// (`engine_collision_box.c:632-641`) for `f32`; then the degenerate-branch tests of the primitive
// colliders: MuJoCo's absolute `mjMINVAL` ones in `f64` (`1e-15 * 1e-15`, no relative part), the
// scale-aware ones in `f32` (see the constants)
impl_real!(f64, 1e-13, 1e-16, 1e-9, 1e-14, 1e-15 * 1e-15, 0.0, 0.0);
impl_real!(f32, 1e-6, 1e-7, 1e-5, 1e-10, f32::EPSILON, 1e-6, 1e-6);

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<R: Real>(v: f64) -> f64 {
        R::from_f64(v).to_f64()
    }

    #[test]
    fn conversions_and_constants() {
        assert_eq!(round_trip::<f64>(0.1), 0.1);
        assert_eq!(round_trip::<f32>(0.5), 0.5);
        assert_eq!(round_trip::<f32>(0.1), f64::from(0.1f32));
        assert_eq!(f64::ZERO + f64::ONE, 1.0);
        assert_eq!(<f32 as Real>::ONE, 1.0f32);
    }

    #[test]
    fn elementary_functions_agree_with_std() {
        fn check<R: Real>() {
            let x = R::from_f64(0.75);
            assert_eq!(x.sqrt().to_f64(), R::from_f64(0.75).sqrt().to_f64());
            assert!((x.sin().to_f64() - 0.75f64.sin()).abs() < 1e-6);
            assert!((x.cos().to_f64() - 0.75f64.cos()).abs() < 1e-6);
            assert!((x.atan2(R::from_f64(-1.0)).to_f64() - 0.75f64.atan2(-1.0)).abs() < 1e-6);
            assert_eq!((-x).abs().to_f64(), x.to_f64());
        }
        check::<f64>();
        check::<f32>();
    }
}
