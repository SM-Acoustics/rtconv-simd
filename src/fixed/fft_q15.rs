//! Pure Rust `#![no_std]` Q1.15 Block-Floating-Point (BFP) real FFT/IFFT for `N = 64, 128, 256`.
//!
//! Zero heap allocation, `#![forbid(unsafe_code)]`, zero C/FFI, FPU-free integer arithmetic.
//! Uses pre-computed Q1.15 twiddle tables (`512 B` ROM) and true block-floating-point scaling:
//!
//! 1. **Input normalization** — the input block is left-shifted until its peak occupies the full
//!    15-bit mantissa (negative block exponent). Without this step a quiet block is transformed
//!    with only a few significant bits and the SNR collapses with signal level, which is exactly
//!    the fixed-point behaviour BFP is meant to avoid (Kalliojärvi & Astola, *Roundoff errors in
//!    block-floating-point systems*, IEEE Trans. SP 44(4), 1996: BFP SNR should be approximately
//!    level-independent).
//! 2. **Per-stage conditional down-scaling** — after each radix-2 stage the block is shifted right
//!    only when the peak exceeds 15 bits (Welch 1969; TI SPRA948), with the shifts accumulated in
//!    the block exponent.
//! 3. **Exact unit twiddles** — `W^0 = 1` and `W^{N/4} = -j` are applied as exact `±32768` in the
//!    `i32` butterfly instead of the `32767` stored in the `i16` ROM table.
//!
//! Stack usage: two `[i32; 256]` work arrays (`2 KiB`) per call.

use crate::fixed::block_float::bfp_downscale_i32_stage;
use crate::fixed::fft_q15_tables::{TWIDDLE_COS_256, TWIDDLE_SIN_256};
use crate::fixed::q15::sat_i16_from_i32;
use crate::{ConvError, FftSize};

const MAX_FIXED_FFT: usize = 256;

#[inline]
const fn log2_fft(size: FftSize) -> u32 {
    match size {
        FftSize::N64 => 6,
        FftSize::N128 => 7,
        FftSize::N256 => 8,
    }
}

#[inline]
const fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    let mut i = 0u32;
    while i < bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
        i += 1;
    }
    r
}

/// Returns `(cos, sin)` of `2*pi*idx/256` in Q1.15 held in `i32`, with the exact unit values
/// `cos(0) = 1` and `sin(pi/2) = 1` represented as `32768` (not representable in the `i16` ROM).
#[inline(always)]
fn twiddle_q15(idx: usize) -> (i32, i32) {
    let c = if idx == 0 {
        32768
    } else {
        TWIDDLE_COS_256[idx] as i32
    };
    let s = if idx == 64 {
        32768
    } else {
        TWIDDLE_SIN_256[idx] as i32
    };
    (c, s)
}

/// Left shift that brings `peak` (`> 0`) into `[16384, 32767]` (full 15-bit mantissa).
/// Returns `0` when `peak` already uses 15 or more bits.
#[inline(always)]
const fn normalize_shift(peak: u32) -> u32 {
    let bits = 32 - peak.leading_zeros();
    15u32.saturating_sub(bits)
}

/// Core in-place radix-2 DIT complex FFT on `re[0..n]`, `im[0..n]` in `i32` with per-stage
/// BFP downscaling. `inverse = false` uses `exp(-j 2*pi*k/N)`; `inverse = true` uses
/// `exp(+j 2*pi*k/N)` (unnormalized; caller subtracts `log2(N)` from the block exponent).
fn cfft_radix2_bfp(re: &mut [i32], im: &mut [i32], size: FftSize, inverse: bool) -> i8 {
    let n = size.len();
    let bits = log2_fft(size);

    // Bit-reversal permutation
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut total_exp: i32 = 0;
    let mut half = 1usize;
    while half < n {
        let step = half << 1;
        let tw_stride = MAX_FIXED_FFT / step;

        let mut base = 0usize;
        while base < n {
            for k in 0..half {
                let (wr, ws) = twiddle_q15(k * tw_stride);
                let wi = if inverse { ws } else { -ws };

                let i0 = base + k;
                let i1 = i0 + half;
                let ar = re[i0];
                let ai = im[i0];
                let br = re[i1];
                let bi = im[i1];

                // Complex multiply W * B in Q1.15 -> i32. Stage inputs satisfy |br|,|bi| <= 32768
                // and |W| <= 1, so |wr*br - wi*bi| <= sqrt(2) * 2^30 < 2^31 (no overflow).
                let tr = (wr * br - wi * bi + 16384) >> 15;
                let ti = (wr * bi + wi * br + 16384) >> 15;

                re[i0] = ar + tr;
                im[i0] = ai + ti;
                re[i1] = ar - tr;
                im[i1] = ai - ti;
            }
            base += step;
        }

        // Per-stage BFP downscaling guarantees |re|, |im| <= 32767 before next stage.
        let shift = bfp_downscale_i32_stage(&mut re[..n], &mut im[..n]);
        total_exp += shift as i32;
        half = step;
    }

    total_exp.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

/// Computes forward real-to-complex Q1.15 BFP FFT (`time_in[0..N]` -> `spec_re[0..N/2+1]`,
/// `spec_im[0..N/2+1]`), returning the spectrum block exponent `exp` such that:
/// `X[k] ≈ ((spec_re[k] + j * spec_im[k]) / 32768.0) * 2^exp`.
pub fn rfft_q15_bfp(
    size: FftSize,
    time_in: &[i16],
    spec_re: &mut [i16],
    spec_im: &mut [i16],
) -> Result<i8, ConvError> {
    let n = size.len();
    let bins = size.spectrum_bins();
    if time_in.len() != n || spec_re.len() != bins || spec_im.len() != bins {
        return Err(ConvError::MismatchedBufferLength {
            expected: n,
            actual: time_in.len(),
        });
    }

    let mut peak = 0u32;
    for &v in time_in {
        peak = peak.max((v as i32).unsigned_abs());
    }
    if peak == 0 {
        spec_re.fill(0);
        spec_im.fill(0);
        return Ok(0);
    }
    // Block normalization: use the full mantissa for quiet blocks (negative exponent).
    let pre_shift = normalize_shift(peak);

    let mut re = [0i32; MAX_FIXED_FFT];
    let mut im = [0i32; MAX_FIXED_FFT];
    for i in 0..n {
        re[i] = (time_in[i] as i32) << pre_shift;
    }

    let stage_exp = cfft_radix2_bfp(&mut re, &mut im, size, false) as i32;
    let exp = (stage_exp - pre_shift as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8;

    for k in 0..bins {
        spec_re[k] = sat_i16_from_i32(re[k]);
        spec_im[k] = sat_i16_from_i32(im[k]);
    }
    // Enforce exact real-input symmetry on DC and Nyquist bins.
    spec_im[0] = 0;
    spec_im[bins - 1] = 0;

    Ok(exp)
}

/// Shared inverse core: validates lengths, rebuilds the Hermitian spectrum with block
/// normalization, runs the inverse BFP CFFT into `re`, and returns the net binary exponent
/// such that `x[i] = re[i] * 2^net_exp / 32768` (or `None` for an all-zero spectrum).
fn irfft_core(
    size: FftSize,
    spec_re: &[i16],
    spec_im: &[i16],
    exponent: i8,
    out_len: usize,
    re: &mut [i32; MAX_FIXED_FFT],
) -> Result<Option<i32>, ConvError> {
    let n = size.len();
    let half = size.half_len();
    let bins = size.spectrum_bins();
    if spec_re.len() != bins || spec_im.len() != bins || out_len != n {
        return Err(ConvError::MismatchedBufferLength {
            expected: n,
            actual: out_len,
        });
    }

    let mut peak = (spec_re[0] as i32)
        .unsigned_abs()
        .max((spec_re[half] as i32).unsigned_abs());
    for k in 1..half {
        peak = peak
            .max((spec_re[k] as i32).unsigned_abs())
            .max((spec_im[k] as i32).unsigned_abs());
    }
    if peak == 0 {
        return Ok(None);
    }
    let pre_shift = normalize_shift(peak);

    let mut im = [0i32; MAX_FIXED_FFT];
    // Reconstruct Hermitian conjugate-symmetric full spectrum (normalized mantissas).
    re[0] = (spec_re[0] as i32) << pre_shift;
    re[half] = (spec_re[half] as i32) << pre_shift;
    for k in 1..half {
        let r = (spec_re[k] as i32) << pre_shift;
        let i = (spec_im[k] as i32) << pre_shift;
        re[k] = r;
        im[k] = i;
        re[n - k] = r;
        im[n - k] = -i;
    }

    let inv_exp = cfft_radix2_bfp(&mut re[..], &mut im, size, true);
    // Exact final exponent after input normalization and 1/N IFFT normalization.
    Ok(Some(
        (exponent as i32) + (inv_exp as i32) - (pre_shift as i32) - (log2_fft(size) as i32),
    ))
}

/// Scales `v * 2^shift` with round-half-up for negative shifts (saturating in `i64`).
#[inline(always)]
fn scale_pow2(v: i64, shift: i32) -> i64 {
    if shift > 0 {
        v.saturating_mul(1i64 << (shift as u32).min(40))
    } else if shift < 0 {
        let s = ((-shift) as u32).min(62);
        v.saturating_add(1i64 << (s - 1)) >> s
    } else {
        v
    }
}

/// Computes normalized inverse complex-to-real Q1.15 BFP FFT (`spec_re[0..N/2+1]`,
/// `spec_im[0..N/2+1]`, `exponent` -> `time_out[0..N]`), applying `1 / N` normalization
/// directly in the block exponent and returning the number of saturated output samples.
pub fn irfft_q15_bfp(
    size: FftSize,
    spec_re: &[i16],
    spec_im: &[i16],
    exponent: i8,
    time_out: &mut [i16],
) -> Result<u64, ConvError> {
    let mut re = [0i32; MAX_FIXED_FFT];
    let Some(net_exp) = irfft_core(size, spec_re, spec_im, exponent, time_out.len(), &mut re)?
    else {
        time_out.fill(0);
        return Ok(0);
    };

    let mut sat_count = 0u64;
    for (dst, &v) in time_out.iter_mut().zip(re.iter()) {
        let scaled = scale_pow2(v as i64, net_exp);
        if scaled > i16::MAX as i64 {
            *dst = i16::MAX;
            sat_count = sat_count.saturating_add(1);
        } else if scaled < i16::MIN as i64 {
            *dst = i16::MIN;
            sat_count = sat_count.saturating_add(1);
        } else {
            *dst = scaled as i16;
        }
    }
    Ok(sat_count)
}

/// Like [`irfft_q15_bfp`] but writes `Q(15 + frac_bits)` samples into `i32`, preserving
/// `frac_bits` extra fractional bits so that several partial convolutions can be summed
/// before a single final rounding to Q1.15. Returns the number of `i32`-saturated samples.
pub(crate) fn irfft_q15_bfp_wide(
    size: FftSize,
    spec_re: &[i16],
    spec_im: &[i16],
    exponent: i8,
    frac_bits: u32,
    time_out: &mut [i32],
) -> Result<u64, ConvError> {
    let mut re = [0i32; MAX_FIXED_FFT];
    let Some(net_exp) = irfft_core(size, spec_re, spec_im, exponent, time_out.len(), &mut re)?
    else {
        time_out.fill(0);
        return Ok(0);
    };

    let mut sat_count = 0u64;
    let shift = net_exp + frac_bits as i32;
    for (dst, &v) in time_out.iter_mut().zip(re.iter()) {
        let scaled = scale_pow2(v as i64, shift);
        if scaled > i32::MAX as i64 {
            *dst = i32::MAX;
            sat_count = sat_count.saturating_add(1);
        } else if scaled < i32::MIN as i64 {
            *dst = i32::MIN;
            sat_count = sat_count.saturating_add(1);
        } else {
            *dst = scaled as i32;
        }
    }
    Ok(sat_count)
}
