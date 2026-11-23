//! Block Floating Point (BFP) scaling and normalization primitives for Q1.15 spectra.
//!
//! A BFP spectrum represents complex values as:
//! `X[k] = ((re[k] + j * im[k]) / 32768.0) * 2^exponent`
//!
//! Maintaining a shared `i8` block exponent per spectrum avoids unconditional `1/2`
//! right-shifts on every radix-2 FFT stage, preserving ~20–25 dB more SNR than fixed-scale
//! Q15 FFT while guaranteeing zero intermediate overflow.

use crate::fixed::q15::sat_i16_from_i32;

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
/// inside `[-32767, +32767]`, returning the right-shift `shift` applied (`>= 0`).
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
        let half = 1i32 << (shift - 1);
        for (r, i) in re.iter_mut().zip(im.iter_mut()) {
            *r = (*r + half) >> shift;
            *i = (*i + half) >> shift;
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
/// `=> out_q15 = round(acc_val / 2^shift)`, `out_exp = base_exp + shift - 15`.
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
        let half = 1i64 << (s - 1);
        for k in 0..acc_re.len() {
            let r = (acc_re[k].saturating_add(half) >> s).clamp(i16::MIN as i64, i16::MAX as i64);
            let i = (acc_im[k].saturating_add(half) >> s).clamp(i16::MIN as i64, i16::MAX as i64);
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
