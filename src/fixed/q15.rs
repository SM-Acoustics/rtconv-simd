//! Q1.15 fixed-point numeric contract, rounding, saturation, and conversion helpers.
//!
//! ## Representation
//! - Signed 16-bit integer `i16` representing values in `[-1.0, 1.0 - 2^-15]`:
//!   `x_real = x_q15 / 32768.0`.
//! - Maximum positive value is `i16::MAX = 32767` (`+0.999969482421875`).
//! - Minimum negative value is `i16::MIN = -32768` (`-1.0`).
//! - Dynamic range is ~`90.3 dB` (`20 * log10(32768)`), 1 LSB = `2^-15 ≈ 3.0518e-5`.
//!
//! ## Rounding & Saturation Policy
//! - Products of two Q1.15 values produce Q2.30 (`i32` / `i64`).
//! - Accumulation uses `i64` (exact for up to `2^33` taps without accumulator overflow).
//! - Final scaling divides by `2^15` with explicit **round-half-up** (`(acc + 16384) >> 15`),
//!   followed by symmetric clamping (`sat_16`) to `[i16::MIN, i16::MAX]`.
//! - Every saturation event can be counted explicitly via [`round_sat_q15_from_i64`].

/// Fractional bit shift for Q1.15 (`15`).
pub const Q15_SHIFT: u32 = 15;

/// Half-LSB rounding bias for Q1.15 (`1 << 14 = 16384`).
pub const Q15_HALF: i64 = 1i64 << (Q15_SHIFT - 1);

/// Scale factor `2^15` as `f32`.
pub const Q15_SCALE_F32: f32 = 32768.0;

/// Scale factor `2^15` as `f64`.
pub const Q15_SCALE_F64: f64 = 32768.0;

/// Scales a Q2.30 `i64` accumulator to Q1.15 with round-half-up and saturation,
/// returning `(result_q15, saturated)`.
#[inline]
pub const fn round_sat_q15_from_i64(acc: i64) -> (i16, bool) {
    let shifted = acc.saturating_add(Q15_HALF) >> Q15_SHIFT;
    if shifted > i16::MAX as i64 {
        (i16::MAX, true)
    } else if shifted < i16::MIN as i64 {
        (i16::MIN, true)
    } else {
        (shifted as i16, false)
    }
}

/// Saturates an `i32` value directly into `i16` (`[i16::MIN, i16::MAX]`).
#[inline]
pub const fn sat_i16_from_i32(x: i32) -> i16 {
    if x > i16::MAX as i32 {
        i16::MAX
    } else if x < i16::MIN as i32 {
        i16::MIN
    } else {
        x as i16
    }
}

/// Multiplies two Q1.15 values with round-half-up and saturation (`(-32768) * (-32768) -> 32767`).
#[inline]
pub const fn mul_q15(a: i16, b: i16) -> i16 {
    let prod = (a as i32) * (b as i32);
    let rounded = (prod + (1i32 << 14)) >> 15;
    sat_i16_from_i32(rounded)
}

/// Converts a normalized `f32` in `[-1.0, 1.0)` to Q1.15 (`i16`) with clamping and rounding.
/// Non-finite inputs map to `0`.
#[inline]
pub fn f32_to_q15(x: f32) -> i16 {
    if !x.is_finite() {
        return 0;
    }
    let scaled = x * Q15_SCALE_F32;
    let rounded = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    if rounded >= i16::MAX as f32 {
        i16::MAX
    } else if rounded <= i16::MIN as f32 {
        i16::MIN
    } else {
        rounded as i16
    }
}

/// Converts a Q1.15 `i16` sample back to `f32`.
#[inline]
pub const fn q15_to_f32(x: i16) -> f32 {
    (x as f32) / Q15_SCALE_F32
}

/// Converts a normalized `f64` in `[-1.0, 1.0)` to Q1.15 (`i16`) with clamping and rounding.
#[inline]
pub fn f64_to_q15(x: f64) -> i16 {
    if !x.is_finite() {
        return 0;
    }
    let scaled = x * Q15_SCALE_F64;
    let rounded = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    if rounded >= i16::MAX as f64 {
        i16::MAX
    } else if rounded <= i16::MIN as f64 {
        i16::MIN
    } else {
        rounded as i16
    }
}

/// Converts a Q1.15 `i16` sample back to `f64`.
#[inline]
pub const fn q15_to_f64(x: i16) -> f64 {
    (x as f64) / Q15_SCALE_F64
}
