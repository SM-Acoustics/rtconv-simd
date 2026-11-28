//! Integer-only numeric conversions shared by the artifact decoder and the automatic layer.
//!
//! Nothing in this module executes a floating-point operation: IEEE 754 values are decomposed
//! from their bit patterns and decimal text is scaled with exact `u128` integer arithmetic, so
//! FPU-less targets (ESP32-C6 `riscv32imac`, Cortex-M0) link no soft-float routine
//! (`__mulsf3`, `__muldf3`, ...) and no `core::num::dec2flt` table for these paths.
//!
//! ## Rounding
//! Every conversion into a fixed-point grid is **round to nearest, ties to even** (RNE):
//! - binary sources follow IEEE Std 754-1985 §4.1 ("if the two nearest representable values
//!   are equally near, the one with its least significant bit zero shall be delivered") and
//!   §5.4 (conversion to integer rounds as specified in §4), with the field decoding of
//!   §3.2.1 (single) / §3.2.2 (double);
//! - decimal sources apply the same rule; ABNT NBR 5891:2014 §2.2–§2.4 state it for decimal
//!   digits, including §2.2's "5 followed by at least one non-zero digit rounds up", which is
//!   what the `sticky` flag below implements.
//!
//! ## Exactness of the decimal path
//! A floor boundary of `|v| * 2^k` lies at `j * 2^-k = j * 5^k * 10^-k`, i.e. on a multiple of
//! `10^-max(k, 0)`. Truncating the decimal expansion at that place (`F = -max(k, 0)`) and
//! recording whether any discarded digit is non-zero (`sticky`) therefore yields the exact
//! floor and an exact "inexact" flag, for an input of any length. Rounding to nearest at `k`
//! is derived from the floor at `k + 1`, so ties are exact as well. For Q1.31 without headroom
//! shift this keeps up to 32 significant digits: a fixed 19-digit (`u64`) mantissa is **not**
//! sufficient, because ties `(2q + 1) * 2^-32` have up to 32 significant digits (for example
//! `3 * 2^-32 = 6.9849193096160888671875e-10` has 23).

/// Every magnitude at or above this value is reported as `MAG_CAP`. It exceeds every
/// destination range used by the crate (Q1.15 and Q1.31 saturate at `2^15` and `2^31`).
pub(crate) const MAG_CAP: u64 = 1 << 40;

/// Bit position (above the integer scale) at which decimal accumulation stops: once the
/// truncated mantissa exceeds `scale << DEC_LIMIT_BITS`, the floor is at least `2^41 > MAG_CAP`.
const DEC_LIMIT_BITS: u32 = 41;

/// Largest supported scaling exponent `k` (Q1.31 + 1 guard bit for rounding).
pub(crate) const MAX_SCALE_EXP: i32 = 32;

/// Smallest supported scaling exponent `k`.
pub(crate) const MIN_SCALE_EXP: i32 = -64;

/// IEEE 754 value decomposed into `(-1)^neg * mant * 2^exp` (integer `mant`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ieee {
    /// `+0` or `-0`.
    Zero,
    /// Finite non-zero value `(-1)^neg * mant * 2^exp`.
    Finite {
        /// Sign bit.
        neg: bool,
        /// Integer significand (implicit bit included for normal numbers).
        mant: u64,
        /// Power of two applied to `mant`.
        exp: i32,
    },
    /// Infinity (`nan == false`) or NaN (`nan == true`) with its sign bit.
    NonFinite {
        /// Sign bit.
        neg: bool,
        /// `true` for NaN.
        nan: bool,
    },
}

impl Ieee {
    /// Decodes IEEE 754-1985 single format fields (§3.2.1): `e = 255` is ∞/NaN, `e = 0` is zero
    /// or denormalized (`2^-126 * 0.f`), otherwise `2^(e-127) * 1.f`.
    #[inline]
    pub(crate) const fn from_f32_bits(bits: u32) -> Self {
        let neg = (bits >> 31) != 0;
        let e = (bits >> 23) & 0xFF;
        let f = (bits & 0x007F_FFFF) as u64;
        if e == 0xFF {
            Self::NonFinite { neg, nan: f != 0 }
        } else if e == 0 {
            if f == 0 {
                Self::Zero
            } else {
                Self::Finite {
                    neg,
                    mant: f,
                    exp: -149,
                }
            }
        } else {
            Self::Finite {
                neg,
                mant: f | (1 << 23),
                exp: e as i32 - 150,
            }
        }
    }

    /// Decodes IEEE 754-1985 double format fields (§3.2.2): `e = 2047` is ∞/NaN, `e = 0` is zero
    /// or denormalized (`2^-1022 * 0.f`), otherwise `2^(e-1023) * 1.f`.
    #[inline]
    pub(crate) const fn from_f64_bits(bits: u64) -> Self {
        let neg = (bits >> 63) != 0;
        let e = ((bits >> 52) & 0x7FF) as i32;
        let f = bits & 0x000F_FFFF_FFFF_FFFF;
        if e == 0x7FF {
            Self::NonFinite { neg, nan: f != 0 }
        } else if e == 0 {
            if f == 0 {
                Self::Zero
            } else {
                Self::Finite {
                    neg,
                    mant: f,
                    exp: -1074,
                }
            }
        } else {
            Self::Finite {
                neg,
                mant: f | (1 << 52),
                exp: e - 1075,
            }
        }
    }
}

/// `(min(floor(mant * 2^exp), MAG_CAP), inexact)`, exact for any `exp`.
#[inline]
pub(crate) const fn dyadic_floor(mant: u64, exp: i32) -> (u64, bool) {
    if mant == 0 {
        return (0, false);
    }
    if exp >= 0 {
        if exp >= 64 {
            return (MAG_CAP, false);
        }
        let v = (mant as u128) << exp;
        let v = if v >= MAG_CAP as u128 {
            MAG_CAP
        } else {
            v as u64
        };
        return (v, false);
    }
    let n = exp.unsigned_abs();
    if n >= 64 {
        return (0, true);
    }
    let q = mant >> n;
    let r = mant & ((1u64 << n) - 1);
    let q = if q >= MAG_CAP { MAG_CAP } else { q };
    (q, r != 0)
}

/// Round to nearest, ties to even, at scale `k` from the floor and inexact flag at `k + 1`.
///
/// With `f2 = floor(x * 2^(k+1))`: `floor(x * 2^k) = f2 >> 1`; the dropped bit `f2 & 1` says
/// whether the fraction is at least one half, and it is exactly one half only when nothing
/// below it is set (`!inexact2`).
#[inline]
pub(crate) const fn rne_from_floor2(f2: u64, inexact2: bool) -> u64 {
    let q = f2 >> 1;
    if (f2 & 1) == 1 && (inexact2 || (q & 1) == 1) {
        q + 1
    } else {
        q
    }
}

/// `RNE(mant * 2^exp)` magnitude (capped at about `MAG_CAP`).
#[inline]
pub(crate) const fn dyadic_rne(mant: u64, exp: i32) -> u64 {
    let (f2, nz) = dyadic_floor(mant, exp.saturating_add(1));
    rne_from_floor2(f2, nz)
}

/// Applies `neg` to a rounded magnitude and clamps to `[min, max]`, reporting saturation.
#[inline]
pub(crate) const fn signed_sat(neg: bool, mag: u64, min: i64, max: i64) -> (i64, bool) {
    // `mag <= MAG_CAP + 1 < 2^41`, so the conversion to `i64` is exact.
    let v = if neg { -(mag as i64) } else { mag as i64 };
    if v > max {
        (max, true)
    } else if v < min {
        (min, true)
    } else {
        (v, false)
    }
}

/// Converts an IEEE value to `RNE(v * 2^k)` clamped to `[min, max]`.
///
/// Returns `None` for NaN; infinities saturate to the matching bound (reported as saturated).
#[inline]
pub(crate) const fn ieee_to_fixed_rne(v: Ieee, k: i32, min: i64, max: i64) -> Option<(i64, bool)> {
    match v {
        Ieee::Zero => Some((0, false)),
        Ieee::Finite { neg, mant, exp } => Some(signed_sat(
            neg,
            dyadic_rne(mant, exp.saturating_add(k)),
            min,
            max,
        )),
        Ieee::NonFinite { nan: true, .. } => None,
        Ieee::NonFinite { neg, nan: false } => Some(if neg { (min, true) } else { (max, true) }),
    }
}

/// `f32` → Q1.31 `RNE(x * 2^(31 - shift))` with symmetric-bound saturation, integer-only via
/// [`f32::to_bits`]. NaN maps to `None`.
#[cfg(any(test, all(feature = "auto", rtconv_no_fpu)))]
#[inline]
pub(crate) const fn f32_to_q31_rne(x: f32, shift: u32) -> Option<(i32, bool)> {
    let k = 31 - (if shift > 31 { 31 } else { shift }) as i32;
    match ieee_to_fixed_rne(
        Ieee::from_f32_bits(x.to_bits()),
        k,
        i32::MIN as i64,
        i32::MAX as i64,
    ) {
        Some((v, sat)) => Some((v as i32, sat)),
        None => None,
    }
}

/// `f32` → Q1.15 `RNE(x * 2^15)` with saturation (NaN → `(0, false)`), integer-only.
#[cfg(any(test, feature = "auto"))]
#[inline]
pub(crate) const fn f32_to_q15_rne_sat(x: f32) -> (i16, bool) {
    match ieee_to_fixed_rne(
        Ieee::from_f32_bits(x.to_bits()),
        15,
        i16::MIN as i64,
        i16::MAX as i64,
    ) {
        Some((v, sat)) => (v as i16, sat),
        None => (0, false),
    }
}

/// Exact `q / 32768` as `f32`, assembled from IEEE 754 single fields without an FPU.
#[cfg(any(test, feature = "auto"))]
#[inline]
pub(crate) const fn q15_to_f32_exact(q: i16) -> f32 {
    if q == 0 {
        return f32::from_bits(0);
    }
    let sign = if q < 0 { 1u32 << 31 } else { 0 };
    let m = (q as i32).unsigned_abs(); // 1..=32768
    let p = 31 - m.leading_zeros(); // MSB position 0..=15
    // value = m * 2^-15 = 1.f * 2^(p - 15)
    let biased = (p as i32 - 15 + 127) as u32;
    let frac = (m << (23 - p)) & 0x007F_FFFF;
    f32::from_bits(sign | (biased << 23) | frac)
}

/// `true` when `|v| > (32767 / 32768) * 2^shift`, i.e. `v * 2^-shift` does not fit Q1.15
/// without clipping. Exact for every finite IEEE value.
#[inline]
pub(crate) const fn ieee_exceeds_q15_headroom(v: Ieee, shift: u32) -> bool {
    match v {
        Ieee::Zero => false,
        Ieee::NonFinite { .. } => true,
        Ieee::Finite { mant, exp, .. } => {
            let (fl, inexact) = dyadic_floor(mant, exp.saturating_add(15 - shift as i32));
            fl > 32767 || (fl == 32767 && inexact)
        }
    }
}

/// `5^k` for `0 <= k <= 32` (`5^32 < 2^75`).
#[inline]
const fn pow5(k: u32) -> u128 {
    let mut p = 1u128;
    let mut i = 0;
    while i < k {
        p *= 5;
        i += 1;
    }
    p
}

/// Largest decimal exponent magnitude tracked; `|exp10| > 10^6` saturates every result anyway.
const EXP10_CLAMP: i64 = 1_000_000;

/// A syntactically valid decimal number borrowed from a text token.
///
/// Grammar: `[+-]? ( D+ ([.,] D*)? | [.,] D+ ) ([eE] [+-]? D+)?`, plus the case-insensitive
/// non-finite spellings `inf`, `infinity` and `nan` (with optional sign) that
/// `core::str::FromStr for f64` also accepts, so the set of accepted lines does not depend on
/// the target. A token containing the decimal comma must not exceed 64 bytes (the limit of
/// the decimal-comma path before this decoder existed).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Decimal<'a> {
    tok: &'a [u8],
    neg: bool,
    non_finite: bool,
    int: &'a [u8],
    frac: &'a [u8],
    exp10: i64,
}

impl<'a> Decimal<'a> {
    /// Parses a full token; `None` if any byte does not belong to the grammar.
    pub(crate) fn parse(tok: &'a [u8]) -> Option<Self> {
        if tok.is_empty() || (tok.len() > 64 && tok.contains(&b',')) {
            return None;
        }
        let (neg, body) = match tok[0] {
            b'-' => (true, &tok[1..]),
            b'+' => (false, &tok[1..]),
            _ => (false, tok),
        };
        if body.eq_ignore_ascii_case(b"inf")
            || body.eq_ignore_ascii_case(b"infinity")
            || body.eq_ignore_ascii_case(b"nan")
        {
            return Some(Self {
                tok,
                neg,
                non_finite: true,
                int: &[],
                frac: &[],
                exp10: 0,
            });
        }
        let digits_end = |s: &[u8], from: usize| {
            let mut i = from;
            while i < s.len() && s[i].is_ascii_digit() {
                i += 1;
            }
            i
        };
        let mut i = digits_end(body, 0);
        let int = &body[..i];
        let mut frac: &[u8] = &[];
        if i < body.len() && (body[i] == b'.' || body[i] == b',') {
            let start = i + 1;
            i = digits_end(body, start);
            frac = &body[start..i];
        }
        if int.is_empty() && frac.is_empty() {
            return None;
        }
        let mut exp10 = 0i64;
        if i < body.len() && (body[i] == b'e' || body[i] == b'E') {
            i += 1;
            let eneg = match body.get(i) {
                Some(b'-') => {
                    i += 1;
                    true
                }
                Some(b'+') => {
                    i += 1;
                    false
                }
                _ => false,
            };
            let start = i;
            while i < body.len() && body[i].is_ascii_digit() {
                if exp10 < EXP10_CLAMP {
                    exp10 = exp10 * 10 + (body[i] - b'0') as i64;
                }
                i += 1;
            }
            if i == start {
                return None;
            }
            if eneg {
                exp10 = -exp10;
            }
        }
        if i != body.len() {
            return None;
        }
        Some(Self {
            tok,
            neg,
            non_finite: false,
            int,
            frac,
            exp10,
        })
    }

    /// `true` for `inf` / `infinity` / `nan` spellings.
    #[inline]
    pub(crate) const fn is_non_finite(&self) -> bool {
        self.non_finite
    }

    /// The token exactly as written (used only by the `f32`/`f64` decoding APIs).
    #[inline]
    pub(crate) const fn token(&self) -> &'a [u8] {
        self.tok
    }

    /// `(min(floor(|v| * 2^k), MAG_CAP), inexact)` for `MIN_SCALE_EXP <= k <= MAX_SCALE_EXP`,
    /// exact for any number of digits (see the module documentation).
    pub(crate) fn floor_scaled(&self, k: i32) -> (u64, bool) {
        debug_assert!(!self.non_finite && (MIN_SCALE_EXP..=MAX_SCALE_EXP).contains(&k));
        // Keep digits whose place is >= f; every floor boundary is a multiple of 10^f.
        let f: i64 = if k > 0 { -(k as i64) } else { 0 };
        let (scale_div, limit) = if k >= 0 {
            let p = pow5(k as u32);
            (p, p << DEC_LIMIT_BITS)
        } else {
            (0, 1u128 << (DEC_LIMIT_BITS + k.unsigned_abs()))
        };
        let mut m: u128 = 0;
        let mut sticky = false;
        let mut place: i64 = self.int.len() as i64 - 1 + self.exp10;
        let mut truncated = false;
        for &c in self.int.iter().chain(self.frac.iter()) {
            let d = (c - b'0') as u128;
            if place >= f {
                if m > limit {
                    return (MAG_CAP, true);
                }
                m = m * 10 + d;
            } else {
                truncated = true;
                if d != 0 {
                    sticky = true;
                    break;
                }
            }
            place -= 1;
        }
        // Implicit trailing zeros down to place f (exponent larger than the written digits).
        if !truncated {
            while place >= f && m != 0 {
                if m > limit {
                    return (MAG_CAP, true);
                }
                m *= 10;
                place -= 1;
            }
        }
        if m > limit {
            return (MAG_CAP, true);
        }
        let (q, rem_nz) = if k >= 0 {
            (m / scale_div, !m.is_multiple_of(scale_div))
        } else {
            let n = k.unsigned_abs();
            (m >> n, (m & ((1u128 << n) - 1)) != 0)
        };
        let q = if q >= MAG_CAP as u128 {
            MAG_CAP
        } else {
            q as u64
        };
        (q, rem_nz || sticky)
    }

    /// `RNE(v * 2^k)` clamped to `[min, max]` (`MIN_SCALE_EXP <= k < MAX_SCALE_EXP`); `None`
    /// for NaN, infinities saturate.
    pub(crate) fn to_fixed_rne(self, k: i32, min: i64, max: i64) -> Option<(i64, bool)> {
        if self.non_finite {
            return None;
        }
        let (f2, nz) = self.floor_scaled(k + 1);
        Some(signed_sat(self.neg, rne_from_floor2(f2, nz), min, max))
    }

    /// `true` when `|v| > (32767 / 32768) * 2^shift` (exact). Non-finite values exceed.
    pub(crate) fn exceeds_q15_headroom(&self, shift: u32) -> bool {
        if self.non_finite {
            return true;
        }
        let (fl, inexact) = self.floor_scaled(15 - shift as i32);
        fl > 32767 || (fl == 32767 && inexact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::format;
    use std::string::String;
    use std::vec::Vec;

    fn q31_ref_f64(x: f64, shift: u32) -> (i32, bool) {
        let s = x * (2.0f64).powi(31 - shift as i32);
        let r = s.round_ties_even();
        if r > i32::MAX as f64 {
            (i32::MAX, true)
        } else if r < i32::MIN as f64 {
            (i32::MIN, true)
        } else {
            (r as i32, false)
        }
    }

    fn dec_q31(tok: &str, shift: u32) -> (i32, bool) {
        let d = Decimal::parse(tok.as_bytes()).unwrap_or_else(|| panic!("parse {tok:?}"));
        let (v, s) = d
            .to_fixed_rne(31 - shift as i32, i32::MIN as i64, i32::MAX as i64)
            .unwrap();
        (v as i32, s)
    }

    /// Exact decimal expansion of `num * 2^-p` (`num < 2^40`, `p <= 40`).
    fn dyadic_decimal(neg: bool, num: u64, p: u32) -> String {
        // num * 2^-p = num * 5^p / 10^p
        let scaled = num as u128 * pow5(p);
        let digits = format!("{scaled:0>width$}", width = p as usize + 1);
        let (ip, fp) = digits.split_at(digits.len() - p as usize);
        format!("{}{ip}.{fp}", if neg { "-" } else { "" })
    }

    #[test]
    fn f32_bits_to_q31_matches_f64_reference_rne() {
        let mut bits = 0u32;
        let mut checked = 0u64;
        loop {
            let x = f32::from_bits(bits);
            for shift in [0u32, 1, 3, 17, 31] {
                let got = f32_to_q31_rne(x, shift);
                if x.is_nan() {
                    assert_eq!(got, None);
                } else {
                    // f32 -> f64 and the power-of-two scaling are exact in f64.
                    assert_eq!(got, Some(q31_ref_f64(x as f64, shift)), "x={x:e} s={shift}");
                }
            }
            checked += 1;
            match bits.checked_add(0x0000_1F3D) {
                Some(b) => bits = b,
                None => break,
            }
        }
        assert!(checked > 100_000);
        // Every exact tie at 2^-32 granularity between 2^-9 and 2^-8 (f32 ulp = 2^-32).
        for m in 0..(1u32 << 15) {
            let x = f32::from_bits(0x3B00_0000 | (m << 8) | 0x80); // force the 2^-32 bit
            assert_eq!(f32_to_q31_rne(x, 0), Some(q31_ref_f64(x as f64, 0)));
        }
    }

    #[test]
    fn f64_bits_to_q31_matches_f64_reference_rne() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..400_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Bias exponents towards the interesting range [2^-40, 2^4].
            let exp = 1023 - 40 + (state >> 58) % 45;
            let bits = (state & 0x800F_FFFF_FFFF_FFFF) | (exp << 52);
            let x = f64::from_bits(bits);
            for shift in [0u32, 2, 31] {
                let v = Ieee::from_f64_bits(bits);
                let (q, s) =
                    ieee_to_fixed_rne(v, 31 - shift as i32, i32::MIN as i64, i32::MAX as i64)
                        .unwrap();
                assert_eq!((q as i32, s), q31_ref_f64(x, shift), "x={x:e}");
            }
        }
    }

    #[test]
    fn q15_io_conversions_are_exact() {
        for q in i16::MIN..=i16::MAX {
            let f = q15_to_f32_exact(q);
            assert_eq!(f.to_bits(), (q as f32 / 32768.0).to_bits(), "q={q}");
            assert_eq!(f32_to_q15_rne_sat(f), (q, false));
        }
        assert_eq!(f32_to_q15_rne_sat(f32::NAN), (0, false));
        assert_eq!(f32_to_q15_rne_sat(f32::INFINITY), (i16::MAX, true));
        assert_eq!(f32_to_q15_rne_sat(f32::NEG_INFINITY), (i16::MIN, true));
        assert_eq!(f32_to_q15_rne_sat(1.0), (i16::MAX, true));
        assert_eq!(f32_to_q15_rne_sat(-1.0), (i16::MIN, false));
        // Ties to even at 2^-16: 0.5 LSB -> 0, 1.5 LSB -> 2, -2.5 LSB -> -2.
        assert_eq!(f32_to_q15_rne_sat(0.5 / 32768.0), (0, false));
        assert_eq!(f32_to_q15_rne_sat(1.5 / 32768.0), (2, false));
        assert_eq!(f32_to_q15_rne_sat(-2.5 / 32768.0), (-2, false));
        // Largest f32 below one half LSB (the old `x * 32768 + 0.5` double-rounded it to 1).
        let below_half = f32::from_bits((0.5f32 / 32768.0).to_bits() - 1);
        assert_eq!(f32_to_q15_rne_sat(below_half), (0, false));
        let mut bits = 0u32;
        loop {
            let x = f32::from_bits(bits);
            if !x.is_nan() {
                let r = (x as f64 * 32768.0).round_ties_even();
                let want = if r > 32767.0 {
                    (i16::MAX, true)
                } else if r < -32768.0 {
                    (i16::MIN, true)
                } else {
                    (r as i16, false)
                };
                assert_eq!(f32_to_q15_rne_sat(x), want, "x={x:e}");
            }
            match bits.checked_add(0x0000_3C1B) {
                Some(b) => bits = b,
                None => break,
            }
        }
    }

    #[test]
    fn decimal_grammar_matches_core_float_parser() {
        // Every token over this alphabet up to length 5 is accepted exactly when
        // `str::parse::<f64>` accepts it (the decimal comma is an extension on top).
        let alphabet = b"09+-.eE";
        let mut tok = Vec::new();
        fn rec(alpha: &[u8], tok: &mut Vec<u8>, depth: usize) {
            if !tok.is_empty() {
                let s = core::str::from_utf8(tok).unwrap();
                let ours = Decimal::parse(tok).is_some();
                let core = s.parse::<f64>().is_ok();
                assert_eq!(ours, core, "{s:?}");
            }
            if depth == 0 {
                return;
            }
            for &c in alpha {
                tok.push(c);
                rec(alpha, tok, depth - 1);
                tok.pop();
            }
        }
        rec(alphabet, &mut tok, 5);
        for s in [
            "inf",
            "-INF",
            "+Infinity",
            "nan",
            "-NaN",
            "1e1000000000000",
            "1E-0",
        ] {
            assert_eq!(
                Decimal::parse(s.as_bytes()).is_some(),
                s.parse::<f64>().is_ok()
            );
        }
        for s in [
            "infin", "1e", "e1", ".", ",", "1,2,3", "1.2.3", "--1", "1e+", "0x10",
        ] {
            assert!(Decimal::parse(s.as_bytes()).is_none(), "{s:?}");
        }
    }

    #[test]
    fn decimal_ties_match_exact_f64_reference() {
        // (2q + 1) * 2^-p is exactly representable in f64 and Rust's parser is correctly
        // rounded, so the f64 reference is exact for these ties.
        let mut state = 0x1234_5678u64;
        for i in 0..20_000u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let q = if i < 64 { i } else { (state >> 33) % (1 << 31) };
            let neg = (state >> 7) & 1 == 1;
            for shift in [0u32, 1, 5] {
                let p = 32 - shift;
                let tie = dyadic_decimal(neg, 2 * q + 1, p);
                let x: f64 = tie.parse().unwrap();
                assert_eq!(dec_q31(&tie, shift), q31_ref_f64(x, shift), "{tie}");
                // Exact tie +/- a perturbation 20+ digits below the last written digit.
                let above = format!("{tie}000000000000000000001");
                let below = format!("{}4{}", &tie[..tie.len() - 1], "9".repeat(22));
                let (va, _) = dec_q31(&above, shift);
                let (vb, sb) = dec_q31(&below, shift);
                let up = q + 1;
                if neg {
                    assert_eq!(va as i64, -(up as i64), "{above}");
                } else if up > i32::MAX as u64 {
                    assert_eq!(va, i32::MAX, "{above}");
                } else {
                    assert_eq!(va as i64, up as i64, "{above}");
                }
                let want_b = if neg { -(q as i64) } else { q as i64 };
                assert_eq!((vb as i64, sb), (want_b, false), "{below}");
            }
        }
    }

    #[test]
    fn nineteen_digit_truncation_counterexample() {
        // 3 * 2^-32: an exact tie between 1 and 2 LSB with 23 significant digits. Truncating to
        // 19 digits (6984919309616088867) with a sticky bit sees "just above 1.4999..." and
        // would round down to 1; the exact decoder rounds the tie to even, 2.
        assert_eq!(dec_q31("0.00000000069849193096160888671875", 0), (2, false));
        assert_eq!(dec_q31("6.9849193096160888671875e-10", 0), (2, false));
        assert_eq!(dec_q31("6,9849193096160888671875E-10", 0), (2, false));
        // 2^-32: tie between 0 and 1 LSB -> 0; any non-zero digit beyond it -> 1.
        assert_eq!(dec_q31("2.3283064365386962890625e-10", 0), (0, false));
        assert_eq!(
            dec_q31("2.32830643653869628906250000000000000000000000001e-10", 0),
            (1, false)
        );
        assert_eq!(
            dec_q31("2.3283064365386962890624999999999999999999e-10", 0),
            (0, false)
        );
        assert_eq!(dec_q31("-2.3283064365386962890625e-10", 0), (0, false));
        assert_eq!(
            dec_q31("-2.32830643653869628906250000000000000000000000001e-10", 0),
            (-1, false)
        );
    }

    #[test]
    fn decimal_saturation_extremes_and_comma() {
        assert_eq!(dec_q31("1", 0), (i32::MAX, true));
        assert_eq!(dec_q31("-1", 0), (i32::MIN, false));
        // Within half an LSB below -1.0 rounds to -1.0 (no clipping); beyond it saturates.
        assert_eq!(
            dec_q31("-1.0000000000000000000000000000001", 0),
            (i32::MIN, false)
        );
        assert_eq!(dec_q31("-1.0000000003", 0), (i32::MIN, true));
        // 1 - 2^-32 is the tie between i32::MAX and 2^31: just below stays, the tie goes to
        // the even 2^31 and therefore saturates.
        assert_eq!(dec_q31("0.99999999976716935634", 0), (i32::MAX, false));
        assert_eq!(
            dec_q31("0.99999999976716935634613037109375", 0),
            (i32::MAX, true)
        );
        assert_eq!(dec_q31("0.5", 0), (1 << 30, false));
        assert_eq!(dec_q31("0,5", 0), (1 << 30, false));
        assert_eq!(dec_q31("-,25", 0), (-(1 << 29), false));
        assert_eq!(dec_q31("5e-1", 0), (1 << 30, false));
        assert_eq!(dec_q31("0.05E+1", 0), (1 << 30, false));
        assert_eq!(
            dec_q31("1.7", 1),
            ((1.7f64 * (1u64 << 30) as f64).round() as i32, false)
        );
        assert_eq!(dec_q31("1e400", 0), (i32::MAX, true));
        assert_eq!(dec_q31("-1e400", 0), (i32::MIN, true));
        assert_eq!(dec_q31("1e-400", 0), (0, false));
        assert_eq!(dec_q31("-1e-400", 0), (0, false));
        assert_eq!(dec_q31("1e999999999999999999999999", 31), (i32::MAX, true));
        assert_eq!(dec_q31("1e-999999999999999999999999", 0), (0, false));
        assert_eq!(dec_q31("0e999999999", 0), (0, false));
        assert_eq!(
            dec_q31("000000000000000000000000000000000.5", 0),
            (1 << 30, false)
        );
        // Over 19 significant digits, f64 reference far from any tie.
        let s = "0.1234567890123456789012345678901234567890";
        let x: f64 = s.parse().unwrap();
        assert_eq!(dec_q31(s, 0), q31_ref_f64(x, 0));
        let s = "-0,9876543210987654321098765";
        let x: f64 = "-0.9876543210987654321098765".parse().unwrap();
        assert_eq!(dec_q31(s, 0), q31_ref_f64(x, 0));
        // Headroom shift: values up to 2^shift.
        assert_eq!(dec_q31("1048575.5", 20), (2_147_482_624, false));
        assert_eq!(dec_q31("1048575.99951171875", 20), (i32::MAX, false));
        assert_eq!(dec_q31("1048576", 20), (i32::MAX, true));
        assert!(
            Decimal::parse(b"nan")
                .unwrap()
                .to_fixed_rne(31, -1, 1)
                .is_none()
        );
    }

    #[test]
    fn decimal_random_against_exact_rational_reference() {
        // value = D / 10^n with D < 10^24: T = D * 2^k / 10^n computed exactly in u128.
        let mut state = 0xDEAD_BEEFu64;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let nd = 1 + (state % 24) as u32;
            let d = (state >> 8) as u128 * 0x1_0000 % 10u128.pow(nd);
            let n = (state >> 40) as u32 % 30;
            let shift = (state >> 50) as u32 % 4;
            let k = 31 - shift;
            let s = {
                let ds = format!("{d:0>width$}", width = n as usize + 1);
                let (a, b) = ds.split_at(ds.len() - n as usize);
                format!("{a}.{b}")
            };
            let num = d << (k + 1);
            let den = 10u128.pow(n);
            let f2 = num / den;
            let nz = !num.is_multiple_of(den);
            let mag = rne_from_floor2(f2.min(MAG_CAP as u128) as u64, nz);
            let want = signed_sat(false, mag, i32::MIN as i64, i32::MAX as i64);
            assert_eq!(
                dec_q31(&s, shift),
                (want.0 as i32, want.1),
                "{s} shift={shift}"
            );
        }
    }

    #[test]
    fn headroom_checks_are_exact() {
        let lim = 32767.0f64 / 32768.0;
        for shift in 0..=4u32 {
            let t = lim * (1u64 << shift) as f64;
            let v = Ieee::from_f64_bits(t.to_bits());
            assert!(!ieee_exceeds_q15_headroom(v, shift));
            let above = Ieee::from_f64_bits(t.to_bits() + 1);
            assert!(ieee_exceeds_q15_headroom(above, shift));
            let s = format!("{t}");
            assert!(
                !Decimal::parse(s.as_bytes())
                    .unwrap()
                    .exceeds_q15_headroom(shift)
            );
            let s = format!("{t}0000000000000000000001");
            assert!(
                Decimal::parse(s.as_bytes())
                    .unwrap()
                    .exceeds_q15_headroom(shift)
            );
        }
        assert!(
            !Decimal::parse(b"-0.999969482421875")
                .unwrap()
                .exceeds_q15_headroom(0)
        );
        assert!(Decimal::parse(b"-1").unwrap().exceeds_q15_headroom(0));
        assert!(!Decimal::parse(b"1.7").unwrap().exceeds_q15_headroom(1));
        assert!(Decimal::parse(b"1.7").unwrap().exceeds_q15_headroom(0));
        assert!(ieee_exceeds_q15_headroom(
            Ieee::from_f32_bits(f32::INFINITY.to_bits()),
            31
        ));
    }
}
