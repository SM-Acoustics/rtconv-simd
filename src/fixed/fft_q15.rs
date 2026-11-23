//! Pure Rust `#![no_std]` Q1.15 Block-Floating-Point (BFP) real FFT/IFFT for `N = 64, 128, 256`.
//!
//! Zero heap allocation, `#![forbid(unsafe_code)]`, zero C/FFI, FPU-free integer arithmetic.
//! Uses pre-computed Q1.15 twiddle tables (`512 B` ROM) and per-stage conditional BFP scaling
//! in `i32` accumulators to prevent intermediate overflow while maximizing SNR.

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
                let tw_idx = k * tw_stride;
                let wr = TWIDDLE_COS_256[tw_idx] as i32;
                let wi = if inverse {
                    TWIDDLE_SIN_256[tw_idx] as i32
                } else {
                    -(TWIDDLE_SIN_256[tw_idx] as i32)
                };

                let i0 = base + k;
                let i1 = i0 + half;
                let ar = re[i0];
                let ai = im[i0];
                let br = re[i1];
                let bi = im[i1];

                // Complex multiply W * B in Q1.15 -> i32 (no overflow: |wr|,|wi| <= 32768, |br|,|bi| <= 32767)
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

    let mut re = [0i32; MAX_FIXED_FFT];
    let mut im = [0i32; MAX_FIXED_FFT];
    for i in 0..n {
        re[i] = time_in[i] as i32;
    }

    let exp = cfft_radix2_bfp(&mut re, &mut im, size, false);

    for k in 0..bins {
        spec_re[k] = sat_i16_from_i32(re[k]);
        spec_im[k] = sat_i16_from_i32(im[k]);
    }
    // Enforce exact real-input symmetry on DC and Nyquist bins.
    spec_im[0] = 0;
    spec_im[bins - 1] = 0;

    Ok(exp)
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
    let n = size.len();
    let half = size.half_len();
    let bins = size.spectrum_bins();
    if spec_re.len() != bins || spec_im.len() != bins || time_out.len() != n {
        return Err(ConvError::MismatchedBufferLength {
            expected: n,
            actual: time_out.len(),
        });
    }

    let mut re = [0i32; MAX_FIXED_FFT];
    let mut im = [0i32; MAX_FIXED_FFT];
    // Reconstruct Hermitian conjugate-symmetric full spectrum.
    re[0] = spec_re[0] as i32;
    im[0] = 0;
    re[half] = spec_re[half] as i32;
    im[half] = 0;
    for k in 1..half {
        re[k] = spec_re[k] as i32;
        im[k] = spec_im[k] as i32;
        re[n - k] = spec_re[k] as i32;
        im[n - k] = -(spec_im[k] as i32);
    }

    let inv_exp = cfft_radix2_bfp(&mut re, &mut im, size, true);
    // Exact final exponent after 1/N IFFT normalization:
    let net_exp: i32 = (exponent as i32) + (inv_exp as i32) - (log2_fft(size) as i32);

    let mut sat_count = 0u64;
    for i in 0..n {
        let v = re[i] as i64;
        let scaled = if net_exp > 0 {
            let s = (net_exp as u32).min(30);
            v.saturating_mul(1i64 << s)
        } else if net_exp < 0 {
            let s = ((-net_exp) as u32).min(62);
            let half_lsb = 1i64 << (s - 1);
            v.saturating_add(half_lsb) >> s
        } else {
            v
        };
        if scaled > i16::MAX as i64 {
            time_out[i] = i16::MAX;
            sat_count = sat_count.saturating_add(1);
        } else if scaled < i16::MIN as i64 {
            time_out[i] = i16::MIN;
            sat_count = sat_count.saturating_add(1);
        } else {
            time_out[i] = scaled as i16;
        }
    }

    Ok(sat_count)
}
