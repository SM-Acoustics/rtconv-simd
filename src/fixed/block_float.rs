//! Block Floating Point (BFP) scaling and normalization primitives for Q1.15 spectra.
//!
//! A BFP spectrum represents complex values as:
//! `X[k] = ((re[k] + j * im[k]) / 32768.0) * 2^exponent`
//!
//! Maintaining a shared `i8` block exponent per spectrum avoids unconditional `1/2`
//! right-shifts on every radix-2 FFT stage, preserving ~20–25 dB more SNR than fixed-scale
//! Q15 FFT while guaranteeing zero intermediate overflow.

use crate::fixed::q15::sat_i16_from_i32;

/// Convergent (round-to-nearest-even) arithmetic right shift for `i32` (`1 <= shift <= 30`).
/// Overflow-free even at `i32::MAX` and `i32::MIN`.
#[inline(always)]
pub(crate) const fn round_shift_even_i32(v: i32, shift: u32) -> i32 {
    let half = 1i32 << (shift - 1);
    let mask = (1i32 << shift) - 1;
    let q = v >> shift;
    let rem = v & mask;
    if rem > half || (rem == half && (q & 1) != 0) {
        q + 1
    } else {
        q
    }
}

/// Convergent (round-to-nearest-even) arithmetic right shift for `i64` (`1 <= shift <= 62`).
/// Overflow-free even at `i64::MAX` and `i64::MIN`.
#[inline(always)]
pub(crate) const fn round_shift_even_i64(v: i64, shift: u32) -> i64 {
    let half = 1i64 << (shift - 1);
    let mask = (1i64 << shift) - 1;
    let q = v >> shift;
    let rem = v & mask;
    if rem > half || (rem == half && (q & 1) != 0) {
        q + 1
    } else {
        q
    }
}

/// Loads a big-endian `i32` packed into two consecutive `i16` words (`zero unsafe`).
#[inline(always)]
pub(crate) fn load_i32_from_i16(buf: &[i16], off: usize) -> i32 {
    (((buf[off] as u16 as u32) << 16) | (buf[off + 1] as u16 as u32)) as i32
}

/// Stores an `i32` into two consecutive `i16` words (`zero unsafe`).
#[inline(always)]
pub(crate) fn store_i32_to_i16(buf: &mut [i16], off: usize, v: i32) {
    let u = v as u32;
    buf[off] = (u >> 16) as u16 as i16;
    buf[off + 1] = u as u16 as i16;
}

/// Loads a big-endian `i64` packed into four consecutive `i16` words (`zero unsafe`).
#[inline(always)]
pub(crate) fn load_i64_from_i16(buf: &[i16], off: usize) -> i64 {
    let b0 = buf[off] as u16 as u64;
    let b1 = buf[off + 1] as u16 as u64;
    let b2 = buf[off + 2] as u16 as u64;
    let b3 = buf[off + 3] as u16 as u64;
    ((b0 << 48) | (b1 << 32) | (b2 << 16) | b3) as i64
}

/// Stores an `i64` into four consecutive `i16` words (`zero unsafe`).
#[inline(always)]
pub(crate) fn store_i64_to_i16(buf: &mut [i16], off: usize, v: i64) {
    let u = v as u64;
    buf[off] = (u >> 48) as u16 as i16;
    buf[off + 1] = (u >> 32) as u16 as i16;
    buf[off + 2] = (u >> 16) as u16 as i16;
    buf[off + 3] = u as u16 as i16;
}

/// Returns the minimum right-shift `s >= 0` such that `peak >> s <= 32767`.
#[inline]
pub const fn excess_shift_i32(peak: u32) -> u32 {
    if peak <= 32767 {
        0
    } else {
        (32 - peak.leading_zeros()) - 15
    }
}

/// Normalizes an `i32` complex buffer `(re[..n], im[..n])` in-place so all values fit
/// inside `[-32767, +32767]` using convergent rounding, returning the right-shift applied (`>= 0`).
#[inline]
pub fn bfp_downscale_i32_stage(re: &mut [i32], im: &mut [i32]) -> i8 {
    let mut peak = 0u32;
    for (&r, &i) in re.iter().zip(im.iter()) {
        let ar = r.unsigned_abs();
        let ai = i.unsigned_abs();
        if ar > peak {
            peak = ar;
        }
        if ai > peak {
            peak = ai;
        }
    }
    let shift = excess_shift_i32(peak);
    if shift > 0 {
        for (r, i) in re.iter_mut().zip(im.iter_mut()) {
            *r = round_shift_even_i32(*r, shift);
            *i = round_shift_even_i32(*i, shift);
        }
    }
    shift as i8
}

/// Normalizes an `i64` Q2.30 spectral accumulator `(acc_re[..bins], acc_im[..bins])` with
/// base exponent `base_exp` into Q1.15 BFP `(out_re[..bins], out_im[..bins])`, returning
/// the resulting block exponent `out_exp`.
///
/// Exact value mapping:
/// `acc_val * 2^(base_exp - 30) ≈ out_q15 * 2^(out_exp - 15)`
/// `=> out_q15 = round_even(acc_val / 2^shift)`, `out_exp = base_exp + shift - 15`.
pub fn bfp_normalize_acc_i64_to_q15(
    acc_re: &[i64],
    acc_im: &[i64],
    base_exp: i8,
    out_re: &mut [i16],
    out_im: &mut [i16],
) -> i8 {
    let mut peak = 0u64;
    for (&r, &i) in acc_re.iter().zip(acc_im.iter()) {
        let ar = r.unsigned_abs();
        let ai = i.unsigned_abs();
        if ar > peak {
            peak = ar;
        }
        if ai > peak {
            peak = ai;
        }
    }
    if peak == 0 {
        out_re.fill(0);
        out_im.fill(0);
        return 0;
    }

    // Choose shift so peak >> shift lands in [16384, 32767] (full Q1.15 precision).
    let bits = 64 - peak.leading_zeros(); // >= 1
    let shift: i32 = (bits as i32) - 15;
    let out_exp = (base_exp as i32) + shift - 15;

    if shift > 0 {
        let s = (shift as u32).min(62);
        for k in 0..acc_re.len() {
            let r = round_shift_even_i64(acc_re[k], s).clamp(i16::MIN as i64, i16::MAX as i64);
            let i = round_shift_even_i64(acc_im[k], s).clamp(i16::MIN as i64, i16::MAX as i64);
            out_re[k] = r as i16;
            out_im[k] = i as i16;
        }
    } else if shift < 0 {
        let s = (-shift as u32).min(62);
        for k in 0..acc_re.len() {
            let r = (acc_re[k] << s).clamp(i16::MIN as i64, i16::MAX as i64);
            let i = (acc_im[k] << s).clamp(i16::MIN as i64, i16::MAX as i64);
            out_re[k] = r as i16;
            out_im[k] = i as i16;
        }
    } else {
        for k in 0..acc_re.len() {
            out_re[k] = sat_i16_from_i32(acc_re[k] as i32);
            out_im[k] = sat_i16_from_i32(acc_im[k] as i32);
        }
    }

    out_exp.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

/// Rounds a Q1.31 tap to Q1.15 (nearest-even) with saturation.
#[inline]
pub(crate) fn q31_to_q15(v: i32) -> i16 {
    let q = round_shift_even_i64(v as i64, 16);
    q.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

/// Source of FIR taps for the fixed-point engines' preparation step.
pub(crate) trait TapSource {
    /// Number of taps.
    fn len(&self) -> usize;
    /// Tap `i` rounded to absolute Q1.15 (direct-form stages).
    fn q15(&self, i: usize) -> i16;
    /// Tap `i` as exact Q1.31 (`value = tap / 2^31`), the input of the partition transforms.
    fn tap_q31(&self, i: usize) -> i32;
}

/// Q1.15 taps used verbatim (bit-exact legacy path).
impl TapSource for &[i16] {
    #[inline]
    fn len(&self) -> usize {
        <[i16]>::len(self)
    }
    #[inline]
    fn q15(&self, i: usize) -> i16 {
        self[i]
    }
    #[inline]
    fn tap_q31(&self, i: usize) -> i32 {
        (self[i] as i32) << 16
    }
}

/// Q1.31 taps accessed by index. FFT partitions are transformed from the exact Q1.31 values
/// and stored with their own block exponent (`crate::fixed::spectral::prepare_partition`).
pub(crate) struct Q31Taps<F: Fn(usize) -> i32> {
    pub len: usize,
    pub tap: F,
}

impl<F: Fn(usize) -> i32> TapSource for Q31Taps<F> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
    #[inline]
    fn q15(&self, i: usize) -> i16 {
        q31_to_q15((self.tap)(i))
    }
    #[inline]
    fn tap_q31(&self, i: usize) -> i32 {
        (self.tap)(i)
    }
}
