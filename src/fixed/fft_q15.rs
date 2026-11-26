//! Pure Rust `#![no_std]` Q1.15 Block-Floating-Point (BFP) real FFT/IFFT for `N = 64 ..= 2048`.
//!
//! Zero heap allocation, `#![forbid(unsafe_code)]`, zero C/FFI, FPU-free integer arithmetic.
//! Uses a single pre-computed Q1.15 quarter-wave twiddle ROM (`1,026 B`) and true
//! block-floating-point scaling:
//!
//! 1. **Input normalization** — the input block is left-shifted until its peak occupies the full
//!    15-bit mantissa (negative block exponent). Without this step a quiet block is transformed
//!    with only a few significant bits and the SNR collapses with signal level, which is exactly
//!    the fixed-point behaviour BFP is meant to avoid (Kalliojärvi & Astola, *Roundoff errors in
//!    block-floating-point systems*, IEEE Trans. SP 44(4), 1996).
//! 2. **Per-stage conditional down-scaling** — after each radix-2 stage the block is shifted right
//!    only when the peak exceeds 15 bits (Welch 1969; TI SPRA948), with convergent rounding
//!    (round-to-nearest-even) and shifts accumulated in the block exponent.
//! 3. **Exact unit twiddles** — `W^0 = 1` and `W^{N/4} = -j` are applied as exact `±32768` in the
//!    `i32` butterfly instead of the `32767` stored in the `i16` ROM table.
//! 4. **Caller-supplied scratch** — [`rfft_q15_bfp_with_scratch`] and
//!    [`irfft_q15_bfp_with_scratch`] operate strictly inside a caller-owned `&mut [i32]` scratch
//!    slice of length `2 * N` (`< 256 B` peak stack for all sizes up to `N = 2048`). The internal
//!    `*_ws` entry points used by `FixedQ15PartitionedConvolver` and `FixedQ15NonUniformConvolver`
//!    pack `re`/`im` into the caller's `&mut [i16]` workspace (`4 * N` `i16` words) with zero
//!    large stack arrays.

use crate::fixed::block_float::{
    bfp_downscale_i32_stage, bfp_downscale_i32_stage_ws, load_i32_from_i16, round_shift_even_i32,
    round_shift_even_i64, store_i32_to_i16,
};
use crate::fixed::fft_q15_tables::{
    FIXED_FFT_ROM_BYTES, MAX_Q15_FFT_SIZE, Q15_TW_QUARTER, TWIDDLE_COS_Q_2048,
};
use crate::fixed::q15::sat_i16_from_i32;
use crate::{ConvError, FftSize};

/// Maximum FFT length supported by the Q1.15 BFP FFT (`2048` samples, block size `1024`).
pub const MAX_FIXED_FFT: usize = MAX_Q15_FFT_SIZE;

const LEGACY_STACK_MAX_FFT: usize = 256;

/// Returns `true` if `size` is supported by the Q1.15 BFP FFT (`N64` through `N2048`).
#[inline]
pub const fn fixed_fft_supports_size(size: FftSize) -> bool {
    size.len() <= MAX_FIXED_FFT
}

/// Required `&mut [i32]` scratch length for [`rfft_q15_bfp_with_scratch`] and
/// [`irfft_q15_bfp_with_scratch`] (`2 * N` `i32` elements: `re[0..N]` + `im[0..N]`).
#[inline]
pub const fn fixed_fft_scratch_len_i32(size: FftSize) -> usize {
    size.len() * 2
}

/// Required `&mut [i16]` workspace scratch length for internal packed `*_ws` transforms
/// (`4 * N` `i16` words = `2 * N` `i32` elements).
#[inline]
pub(crate) const fn fixed_fft_scratch_len_i16(size: FftSize) -> usize {
    size.len() * 4
}

/// Exact read-only twiddle table size in bytes for the Q1.15 FFT (`1,026 B`).
#[inline]
pub const fn fixed_fft_rom_bytes() -> usize {
    FIXED_FFT_ROM_BYTES
}

#[inline]
const fn bit_reverse(x: usize, bits: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - bits)
}

/// Returns `(cos, sin)` of `2*pi*idx/2048` in Q1.15 held in `i32` (`idx in 0..1024`),
/// with exact unit values `cos(0) = 1` and `sin(pi/2) = 1` represented as `32768`.
#[inline(always)]
fn twiddle_q15(idx: usize) -> (i32, i32) {
    if idx == 0 {
        (32768, 0)
    } else if idx == Q15_TW_QUARTER {
        (0, 32768)
    } else if idx < Q15_TW_QUARTER {
        (
            TWIDDLE_COS_Q_2048[idx] as i32,
            TWIDDLE_COS_Q_2048[Q15_TW_QUARTER - idx] as i32,
        )
    } else {
        let rem = idx - Q15_TW_QUARTER;
        (
            -(TWIDDLE_COS_Q_2048[Q15_TW_QUARTER - rem] as i32),
            TWIDDLE_COS_Q_2048[rem] as i32,
        )
    }
}

/// Left shift that brings `peak` (`> 0`) into `[16384, 32767]` (full 15-bit mantissa).
/// Returns `0` when `peak` already uses 15 or more bits.
#[inline(always)]
const fn normalize_shift(peak: u32) -> u32 {
    let bits = 32 - peak.leading_zeros();
    15u32.saturating_sub(bits)
}

/// Scales `v * 2^shift` with convergent rounding for negative shifts (saturating in `i64`).
#[inline(always)]
fn scale_pow2(v: i64, shift: i32) -> i64 {
    if shift > 0 {
        v.saturating_mul(1i64 << (shift as u32).min(40))
    } else if shift < 0 {
        let s = ((-shift) as u32).min(62);
        round_shift_even_i64(v, s)
    } else {
        v
    }
}

/// Executes one radix-2 DIT butterfly stage (`half = 1, 2, ..., n/2`) on `re[0..n]`, `im[0..n]`
/// plus per-stage BFP downscaling, returning the BFP right-shift applied (`0..=2`).
#[inline]
fn cfft_radix2_bfp_stage(
    re: &mut [i32],
    im: &mut [i32],
    n: usize,
    half: usize,
    inverse: bool,
) -> i32 {
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

            // Complex multiply W * B in Q1.15 -> i32 with convergent rounding.
            // Stage inputs satisfy |br|,|bi| <= 32768 and |W| <= 1, so
            // |wr*br - wi*bi| <= sqrt(2) * 2^30 < 2^31 (no i32 overflow).
            let tr = round_shift_even_i32(wr * br - wi * bi, 15);
            let ti = round_shift_even_i32(wr * bi + wi * br, 15);

            re[i0] = ar + tr;
            im[i0] = ai + ti;
            re[i1] = ar - tr;
            im[i1] = ai - ti;
        }
        base += step;
    }

    // Per-stage BFP downscaling guarantees |re|, |im| <= 32767 before next stage.
    bfp_downscale_i32_stage(&mut re[..n], &mut im[..n]) as i32
}

/// Core in-place radix-2 DIT complex FFT on `re[0..n]`, `im[0..n]` in `i32` with per-stage
/// BFP downscaling. `inverse = false` uses `exp(-j 2*pi*k/N)`; `inverse = true` uses
/// `exp(+j 2*pi*k/N)` (unnormalized; caller subtracts `log2(N)` from the block exponent).
fn cfft_radix2_bfp(re: &mut [i32], im: &mut [i32], size: FftSize, inverse: bool) -> i8 {
    let n = size.len();
    let bits = size.log2();

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
        total_exp += cfft_radix2_bfp_stage(re, im, n, half, inverse);
        half <<= 1;
    }

    total_exp.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

/// Workspace-packed (`i16`-backed `i32` pairs) single radix-2 DIT stage + BFP downscale,
/// bit-identical to [`cfft_radix2_bfp_stage`].
#[inline]
fn cfft_radix2_bfp_ws_stage(
    re_ws: &mut [i16],
    im_ws: &mut [i16],
    n: usize,
    half: usize,
    inverse: bool,
) -> i32 {
    let step = half << 1;
    let tw_stride = MAX_FIXED_FFT / step;

    let mut base = 0usize;
    while base < n {
        for k in 0..half {
            let (wr, ws) = twiddle_q15(k * tw_stride);
            let wi = if inverse { ws } else { -ws };

            let o0 = (base + k) * 2;
            let o1 = (base + k + half) * 2;
            let ar = load_i32_from_i16(re_ws, o0);
            let ai = load_i32_from_i16(im_ws, o0);
            let br = load_i32_from_i16(re_ws, o1);
            let bi = load_i32_from_i16(im_ws, o1);

            let tr = round_shift_even_i32(wr * br - wi * bi, 15);
            let ti = round_shift_even_i32(wr * bi + wi * br, 15);

            store_i32_to_i16(re_ws, o0, ar + tr);
            store_i32_to_i16(im_ws, o0, ai + ti);
            store_i32_to_i16(re_ws, o1, ar - tr);
            store_i32_to_i16(im_ws, o1, ai - ti);
        }
        base += step;
    }

    bfp_downscale_i32_stage_ws(re_ws, im_ws, n) as i32
}

/// Workspace-packed (`i16`-backed `i32` pairs) bit-identical counterpart of [`cfft_radix2_bfp`].
fn cfft_radix2_bfp_ws(re_ws: &mut [i16], im_ws: &mut [i16], size: FftSize, inverse: bool) -> i8 {
    let n = size.len();
    let bits = size.log2();

    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            let oi = i * 2;
            let oj = j * 2;
            let ri = load_i32_from_i16(re_ws, oi);
            let rj = load_i32_from_i16(re_ws, oj);
            store_i32_to_i16(re_ws, oi, rj);
            store_i32_to_i16(re_ws, oj, ri);

            let ii = load_i32_from_i16(im_ws, oi);
            let ij = load_i32_from_i16(im_ws, oj);
            store_i32_to_i16(im_ws, oi, ij);
            store_i32_to_i16(im_ws, oj, ii);
        }
    }

    let mut total_exp: i32 = 0;
    let mut half = 1usize;
    while half < n {
        total_exp += cfft_radix2_bfp_ws_stage(re_ws, im_ws, n, half, inverse);
        half <<= 1;
    }

    total_exp.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

/// Computes forward real-to-complex Q1.15 BFP FFT (`N = 64 ..= 2048`) inside caller-owned
/// `scratch` (`scratch.len() >= 2 * N`, zero large stack arrays), returning the block exponent
/// `exp` such that `X[k] ≈ ((spec_re[k] + j * spec_im[k]) / 32768.0) * 2^exp`.
pub fn rfft_q15_bfp_with_scratch(
    size: FftSize,
    time_in: &[i16],
    spec_re: &mut [i16],
    spec_im: &mut [i16],
    scratch: &mut [i32],
) -> Result<i8, ConvError> {
    let n = size.len();
    if !fixed_fft_supports_size(size) {
        return Err(ConvError::UnsupportedFftSize { fft_size: n });
    }
    let bins = size.spectrum_bins();
    if time_in.len() != n || spec_re.len() != bins || spec_im.len() != bins {
        return Err(ConvError::MismatchedBufferLength {
            expected: n,
            actual: time_in.len(),
        });
    }
    let req_scratch = fixed_fft_scratch_len_i32(size);
    if scratch.len() < req_scratch {
        return Err(ConvError::WorkspaceTooSmall {
            required: req_scratch,
            provided: scratch.len(),
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
    let pre_shift = normalize_shift(peak);

    let (re, im) = scratch[..req_scratch].split_at_mut(n);
    for i in 0..n {
        re[i] = (time_in[i] as i32) << pre_shift;
    }
    im.fill(0);

    let stage_exp = cfft_radix2_bfp(re, im, size, false) as i32;
    let exp = (stage_exp - pre_shift as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8;

    for k in 0..bins {
        spec_re[k] = sat_i16_from_i32(re[k]);
        spec_im[k] = sat_i16_from_i32(im[k]);
    }
    spec_im[0] = 0;
    spec_im[bins - 1] = 0;

    Ok(exp)
}

/// Computes normalized inverse complex-to-real Q1.15 BFP FFT (`N = 64 ..= 2048`) inside
/// caller-owned `scratch` (`scratch.len() >= 2 * N`, zero large stack arrays), applying `1 / N`
/// normalization in the block exponent and returning the number of saturated output samples.
pub fn irfft_q15_bfp_with_scratch(
    size: FftSize,
    spec_re: &[i16],
    spec_im: &[i16],
    exponent: i8,
    time_out: &mut [i16],
    scratch: &mut [i32],
) -> Result<u64, ConvError> {
    let n = size.len();
    if !fixed_fft_supports_size(size) {
        return Err(ConvError::UnsupportedFftSize { fft_size: n });
    }
    let half = size.half_len();
    let bins = size.spectrum_bins();
    if spec_re.len() != bins || spec_im.len() != bins || time_out.len() != n {
        return Err(ConvError::MismatchedBufferLength {
            expected: n,
            actual: time_out.len(),
        });
    }
    let req_scratch = fixed_fft_scratch_len_i32(size);
    if scratch.len() < req_scratch {
        return Err(ConvError::WorkspaceTooSmall {
            required: req_scratch,
            provided: scratch.len(),
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
        time_out.fill(0);
        return Ok(0);
    }
    let pre_shift = normalize_shift(peak);

    let (re, im) = scratch[..req_scratch].split_at_mut(n);
    im[0] = 0;
    im[half] = 0;
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

    let inv_exp = cfft_radix2_bfp(re, im, size, true);
    let net_exp = (exponent as i32) + (inv_exp as i32) - (pre_shift as i32) - (size.log2() as i32);

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

/// Stack-backed convenience wrapper for `N64`/`N128`/`N256` (`2 KiB` stack).
///
/// For `N512..=N2048` or MCU ISRs with tight stack budgets, use [`rfft_q15_bfp_with_scratch`].
pub fn rfft_q15_bfp(
    size: FftSize,
    time_in: &[i16],
    spec_re: &mut [i16],
    spec_im: &mut [i16],
) -> Result<i8, ConvError> {
    if size.len() > LEGACY_STACK_MAX_FFT {
        return Err(ConvError::UnsupportedFftSize {
            fft_size: size.len(),
        });
    }
    let mut scratch = [0i32; LEGACY_STACK_MAX_FFT * 2];
    rfft_q15_bfp_with_scratch(size, time_in, spec_re, spec_im, &mut scratch)
}

/// Stack-backed convenience wrapper for `N64`/`N128`/`N256` (`2 KiB` stack).
///
/// For `N512..=N2048` or MCU ISRs with tight stack budgets, use [`irfft_q15_bfp_with_scratch`].
pub fn irfft_q15_bfp(
    size: FftSize,
    spec_re: &[i16],
    spec_im: &[i16],
    exponent: i8,
    time_out: &mut [i16],
) -> Result<u64, ConvError> {
    if size.len() > LEGACY_STACK_MAX_FFT {
        return Err(ConvError::UnsupportedFftSize {
            fft_size: size.len(),
        });
    }
    let mut scratch = [0i32; LEGACY_STACK_MAX_FFT * 2];
    irfft_q15_bfp_with_scratch(size, spec_re, spec_im, exponent, time_out, &mut scratch)
}

/// Internal forward RFFT operating inside a caller `&mut [i16]` workspace scratch (`>= 4 * N`
/// `i16` words). Bit-identical to [`rfft_q15_bfp_with_scratch`] with zero large stack arrays.
pub(crate) fn rfft_q15_bfp_ws(
    size: FftSize,
    time_in: &[i16],
    spec_re: &mut [i16],
    spec_im: &mut [i16],
    ws_scratch: &mut [i16],
) -> Result<i8, ConvError> {
    let n = size.len();
    if !fixed_fft_supports_size(size) {
        return Err(ConvError::UnsupportedFftSize { fft_size: n });
    }
    let bins = size.spectrum_bins();
    if time_in.len() != n || spec_re.len() != bins || spec_im.len() != bins {
        return Err(ConvError::MismatchedBufferLength {
            expected: n,
            actual: time_in.len(),
        });
    }
    let req = fixed_fft_scratch_len_i16(size);
    if ws_scratch.len() < req {
        return Err(ConvError::WorkspaceTooSmall {
            required: req,
            provided: ws_scratch.len(),
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
    let pre_shift = normalize_shift(peak);

    let (re_ws, im_ws) = ws_scratch[..req].split_at_mut(2 * n);
    for (i, &v) in time_in.iter().enumerate() {
        store_i32_to_i16(re_ws, i * 2, (v as i32) << pre_shift);
    }
    im_ws.fill(0);

    let stage_exp = cfft_radix2_bfp_ws(re_ws, im_ws, size, false) as i32;
    let exp = (stage_exp - pre_shift as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8;

    for k in 0..bins {
        spec_re[k] = sat_i16_from_i32(load_i32_from_i16(re_ws, k * 2));
        spec_im[k] = sat_i16_from_i32(load_i32_from_i16(im_ws, k * 2));
    }
    spec_im[0] = 0;
    spec_im[bins - 1] = 0;

    Ok(exp)
}

/// Shared workspace-backed inverse core. On `Ok(Some(net_exp))`, the unscaled inverse samples live
/// as packed `i32`s in `ws_scratch[0..2*n]` such that `x[i] = re[i] * 2^net_exp / 32768`.
fn irfft_core_ws(
    size: FftSize,
    spec_re: &[i16],
    spec_im: &[i16],
    exponent: i8,
    ws_scratch: &mut [i16],
) -> Result<Option<i32>, ConvError> {
    let n = size.len();
    if !fixed_fft_supports_size(size) {
        return Err(ConvError::UnsupportedFftSize { fft_size: n });
    }
    let half = size.half_len();
    let bins = size.spectrum_bins();
    if spec_re.len() != bins || spec_im.len() != bins {
        return Err(ConvError::MismatchedBufferLength {
            expected: bins,
            actual: spec_re.len().min(spec_im.len()),
        });
    }
    let req = fixed_fft_scratch_len_i16(size);
    if ws_scratch.len() < req {
        return Err(ConvError::WorkspaceTooSmall {
            required: req,
            provided: ws_scratch.len(),
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

    let (re_ws, im_ws) = ws_scratch[..req].split_at_mut(2 * n);
    store_i32_to_i16(re_ws, 0, (spec_re[0] as i32) << pre_shift);
    store_i32_to_i16(im_ws, 0, 0);
    store_i32_to_i16(re_ws, half * 2, (spec_re[half] as i32) << pre_shift);
    store_i32_to_i16(im_ws, half * 2, 0);
    for k in 1..half {
        let r = (spec_re[k] as i32) << pre_shift;
        let i = (spec_im[k] as i32) << pre_shift;
        store_i32_to_i16(re_ws, k * 2, r);
        store_i32_to_i16(im_ws, k * 2, i);
        store_i32_to_i16(re_ws, (n - k) * 2, r);
        store_i32_to_i16(im_ws, (n - k) * 2, -i);
    }

    let inv_exp = cfft_radix2_bfp_ws(re_ws, im_ws, size, true);
    Ok(Some(
        (exponent as i32) + (inv_exp as i32) - (pre_shift as i32) - (size.log2() as i32),
    ))
}

/// Internal inverse RFFT operating inside caller `&mut [i16]` workspace scratch (`>= 4 * N`
/// `i16` words), writing `time_out[0..N]` in Q1.15.
pub(crate) fn irfft_q15_bfp_ws(
    size: FftSize,
    spec_re: &[i16],
    spec_im: &[i16],
    exponent: i8,
    time_out: &mut [i16],
    ws_scratch: &mut [i16],
) -> Result<u64, ConvError> {
    let n = size.len();
    if time_out.len() != n {
        return Err(ConvError::MismatchedBufferLength {
            expected: n,
            actual: time_out.len(),
        });
    }
    let Some(net_exp) = irfft_core_ws(size, spec_re, spec_im, exponent, ws_scratch)? else {
        time_out.fill(0);
        return Ok(0);
    };

    let mut sat_count = 0u64;
    for (i, dst) in time_out.iter_mut().enumerate() {
        let v = load_i32_from_i16(ws_scratch, i * 2);
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

/// Number of bounded-cost radix-2 BFP stages for time-distributed Q1.15 FFT/IFFT (`log2(N)`).
#[inline]
pub const fn fixed_fft_staged_steps(size: FftSize) -> usize {
    size.log2() as usize
}

/// Compact per-stage state carried across calls to [`rfft_q15_bfp_step_with_scratch`] or
/// [`irfft_q15_bfp_step_with_scratch`] (zero heap, 8 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FixedStagedFftState {
    exp_acc: i32,
    is_zero: bool,
}

impl FixedStagedFftState {
    /// Creates a clean initial staged BFP transform state.
    #[inline]
    pub const fn new() -> Self {
        Self {
            exp_acc: 0,
            is_zero: false,
        }
    }
}

/// Executes one bounded-cost forward Q1.15 BFP RFFT stage (`step` in `0..fixed_fft_staged_steps(size)`)
/// inside caller-owned `scratch` (`len >= 2 * N`).
///
/// Produces bit-identical output and block exponent to [`rfft_q15_bfp_with_scratch`].
/// Returns `Ok(Some(exp))` on the final step (`step + 1 == fixed_fft_staged_steps(size)`) and
/// `Ok(None)` on intermediate steps.
pub fn rfft_q15_bfp_step_with_scratch(
    size: FftSize,
    step: usize,
    time_in: &[i16],
    spec_re: &mut [i16],
    spec_im: &mut [i16],
    scratch: &mut [i32],
    state: &mut FixedStagedFftState,
) -> Result<Option<i8>, ConvError> {
    let n = size.len();
    if !fixed_fft_supports_size(size) {
        return Err(ConvError::UnsupportedFftSize { fft_size: n });
    }
    let total_steps = fixed_fft_staged_steps(size);
    if step >= total_steps {
        return Err(ConvError::InvalidStateTransition);
    }
    let bins = size.spectrum_bins();
    if spec_re.len() != bins || spec_im.len() != bins {
        return Err(ConvError::MismatchedBufferLength {
            expected: bins,
            actual: spec_re.len().min(spec_im.len()),
        });
    }
    let req_scratch = fixed_fft_scratch_len_i32(size);
    if scratch.len() < req_scratch {
        return Err(ConvError::WorkspaceTooSmall {
            required: req_scratch,
            provided: scratch.len(),
        });
    }

    let (re, im) = scratch[..req_scratch].split_at_mut(n);
    if step == 0 {
        if time_in.len() != n {
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
            state.is_zero = true;
            state.exp_acc = 0;
            spec_re.fill(0);
            spec_im.fill(0);
        } else {
            state.is_zero = false;
            let pre_shift = normalize_shift(peak);
            let bits = size.log2();
            for (i, dst) in re.iter_mut().enumerate() {
                let src = bit_reverse(i, bits);
                *dst = (time_in[src] as i32) << pre_shift;
            }
            im.fill(0);
            let shift = cfft_radix2_bfp_stage(re, im, n, 1, false);
            state.exp_acc = shift - pre_shift as i32;
        }
    } else if !state.is_zero {
        let half = 1usize << step;
        state.exp_acc += cfft_radix2_bfp_stage(re, im, n, half, false);
    }

    if step + 1 == total_steps {
        if state.is_zero {
            return Ok(Some(0));
        }
        let exp = state.exp_acc.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        for k in 0..bins {
            spec_re[k] = sat_i16_from_i32(re[k]);
            spec_im[k] = sat_i16_from_i32(im[k]);
        }
        spec_im[0] = 0;
        spec_im[bins - 1] = 0;
        Ok(Some(exp))
    } else {
        Ok(None)
    }
}

/// Executes one bounded-cost normalized inverse Q1.15 BFP RFFT stage
/// (`step` in `0..fixed_fft_staged_steps(size)`) inside caller-owned `scratch` (`len >= 2 * N`).
///
/// Produces bit-identical output and saturation count to [`irfft_q15_bfp_with_scratch`].
/// Returns `Ok(Some(sat_count))` on the final step and `Ok(None)` on intermediate steps.
#[allow(clippy::too_many_arguments)]
pub fn irfft_q15_bfp_step_with_scratch(
    size: FftSize,
    step: usize,
    spec_re: &[i16],
    spec_im: &[i16],
    exponent: i8,
    time_out: &mut [i16],
    scratch: &mut [i32],
    state: &mut FixedStagedFftState,
) -> Result<Option<u64>, ConvError> {
    let n = size.len();
    if !fixed_fft_supports_size(size) {
        return Err(ConvError::UnsupportedFftSize { fft_size: n });
    }
    let total_steps = fixed_fft_staged_steps(size);
    if step >= total_steps {
        return Err(ConvError::InvalidStateTransition);
    }
    let half = size.half_len();
    let bins = size.spectrum_bins();
    let req_scratch = fixed_fft_scratch_len_i32(size);
    if scratch.len() < req_scratch {
        return Err(ConvError::WorkspaceTooSmall {
            required: req_scratch,
            provided: scratch.len(),
        });
    }

    let (re, im) = scratch[..req_scratch].split_at_mut(n);
    if step == 0 {
        if spec_re.len() != bins || spec_im.len() != bins {
            return Err(ConvError::MismatchedBufferLength {
                expected: bins,
                actual: spec_re.len().min(spec_im.len()),
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
            state.is_zero = true;
            state.exp_acc = 0;
        } else {
            state.is_zero = false;
            let pre_shift = normalize_shift(peak);
            let bits = size.log2();
            re[0] = (spec_re[0] as i32) << pre_shift;
            im[0] = 0;
            let br_half = bit_reverse(half, bits);
            re[br_half] = (spec_re[half] as i32) << pre_shift;
            im[br_half] = 0;
            for k in 1..half {
                let r = (spec_re[k] as i32) << pre_shift;
                let i = (spec_im[k] as i32) << pre_shift;
                let br_k = bit_reverse(k, bits);
                let br_nk = bit_reverse(n - k, bits);
                re[br_k] = r;
                im[br_k] = i;
                re[br_nk] = r;
                im[br_nk] = -i;
            }
            let shift = cfft_radix2_bfp_stage(re, im, n, 1, true);
            state.exp_acc = (exponent as i32) + shift - (pre_shift as i32) - (size.log2() as i32);
        }
    } else if !state.is_zero {
        let stage_half = 1usize << step;
        state.exp_acc += cfft_radix2_bfp_stage(re, im, n, stage_half, true);
    }

    if step + 1 == total_steps {
        if time_out.len() != n {
            return Err(ConvError::MismatchedBufferLength {
                expected: n,
                actual: time_out.len(),
            });
        }
        if state.is_zero {
            time_out.fill(0);
            return Ok(Some(0));
        }
        let net_exp = state.exp_acc;
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
        Ok(Some(sat_count))
    } else {
        Ok(None)
    }
}

/// Internal staged forward RFFT operating in-place inside a stage's `i16`-packed workspace buffer
/// (`ws_buf.len() >= 4 * N`, zero extra per-stage RAM). Bit-identical to [`rfft_q15_bfp_ws`].
#[cfg(feature = "fixed-nonuniform")]
pub(crate) fn rfft_q15_bfp_ws_step(
    size: FftSize,
    step: usize,
    time_in: &[i16],
    spec_re: &mut [i16],
    spec_im: &mut [i16],
    ws_buf: &mut [i16],
    state: &mut FixedStagedFftState,
) -> Result<Option<i8>, ConvError> {
    let n = size.len();
    if !fixed_fft_supports_size(size) {
        return Err(ConvError::UnsupportedFftSize { fft_size: n });
    }
    let total_steps = fixed_fft_staged_steps(size);
    if step >= total_steps {
        return Err(ConvError::InvalidStateTransition);
    }
    let bins = size.spectrum_bins();
    let req = fixed_fft_scratch_len_i16(size);
    if ws_buf.len() < req {
        return Err(ConvError::WorkspaceTooSmall {
            required: req,
            provided: ws_buf.len(),
        });
    }

    let (re_ws, im_ws) = ws_buf[..req].split_at_mut(2 * n);
    if step == 0 {
        if time_in.len() != n {
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
            state.is_zero = true;
            state.exp_acc = 0;
            spec_re.fill(0);
            spec_im.fill(0);
        } else {
            state.is_zero = false;
            let pre_shift = normalize_shift(peak);
            let bits = size.log2();
            for i in 0..n {
                let src = bit_reverse(i, bits);
                store_i32_to_i16(re_ws, i * 2, (time_in[src] as i32) << pre_shift);
            }
            im_ws.fill(0);
            let shift = cfft_radix2_bfp_ws_stage(re_ws, im_ws, n, 1, false);
            state.exp_acc = shift - pre_shift as i32;
        }
    } else if !state.is_zero {
        let half = 1usize << step;
        state.exp_acc += cfft_radix2_bfp_ws_stage(re_ws, im_ws, n, half, false);
    }

    if step + 1 == total_steps {
        if state.is_zero {
            return Ok(Some(0));
        }
        let exp = state.exp_acc.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        for k in 0..bins {
            spec_re[k] = sat_i16_from_i32(load_i32_from_i16(re_ws, k * 2));
            spec_im[k] = sat_i16_from_i32(load_i32_from_i16(im_ws, k * 2));
        }
        spec_im[0] = 0;
        spec_im[bins - 1] = 0;
        Ok(Some(exp))
    } else {
        Ok(None)
    }
}

/// Internal staged wide inverse RFFT operating in-place inside a stage's `i16`-packed workspace
/// buffer (`ws_buf.len() >= 4 * N`). On the final step, emits the `N / 2` valid OLS second-half
/// samples in `Q(15 + frac_bits)` via `emit(k, sample)` and returns the saturation count.
#[cfg(feature = "fixed-nonuniform")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn irfft_q15_bfp_wide_ws_step<Emit: FnMut(usize, i32)>(
    size: FftSize,
    step: usize,
    spec_re: &[i16],
    spec_im: &[i16],
    exponent: i8,
    frac_bits: u32,
    ws_buf: &mut [i16],
    state: &mut FixedStagedFftState,
    mut emit: Emit,
) -> Result<u64, ConvError> {
    let n = size.len();
    if !fixed_fft_supports_size(size) {
        return Err(ConvError::UnsupportedFftSize { fft_size: n });
    }
    let total_steps = fixed_fft_staged_steps(size);
    if step >= total_steps {
        return Err(ConvError::InvalidStateTransition);
    }
    let half = size.half_len();
    let req = fixed_fft_scratch_len_i16(size);
    if ws_buf.len() < req {
        return Err(ConvError::WorkspaceTooSmall {
            required: req,
            provided: ws_buf.len(),
        });
    }

    let (re_ws, im_ws) = ws_buf[..req].split_at_mut(2 * n);
    if step == 0 {
        let bins = size.spectrum_bins();
        if spec_re.len() != bins || spec_im.len() != bins {
            return Err(ConvError::MismatchedBufferLength {
                expected: bins,
                actual: spec_re.len().min(spec_im.len()),
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
            state.is_zero = true;
            state.exp_acc = 0;
        } else {
            state.is_zero = false;
            let pre_shift = normalize_shift(peak);
            let bits = size.log2();
            store_i32_to_i16(re_ws, 0, (spec_re[0] as i32) << pre_shift);
            store_i32_to_i16(im_ws, 0, 0);
            let br_half = bit_reverse(half, bits);
            store_i32_to_i16(re_ws, br_half * 2, (spec_re[half] as i32) << pre_shift);
            store_i32_to_i16(im_ws, br_half * 2, 0);
            for k in 1..half {
                let r = (spec_re[k] as i32) << pre_shift;
                let i = (spec_im[k] as i32) << pre_shift;
                let br_k = bit_reverse(k, bits);
                let br_nk = bit_reverse(n - k, bits);
                store_i32_to_i16(re_ws, br_k * 2, r);
                store_i32_to_i16(im_ws, br_k * 2, i);
                store_i32_to_i16(re_ws, br_nk * 2, r);
                store_i32_to_i16(im_ws, br_nk * 2, -i);
            }
            let shift = cfft_radix2_bfp_ws_stage(re_ws, im_ws, n, 1, true);
            state.exp_acc = (exponent as i32) + shift - (pre_shift as i32) - (size.log2() as i32);
        }
    } else if !state.is_zero {
        let stage_half = 1usize << step;
        state.exp_acc += cfft_radix2_bfp_ws_stage(re_ws, im_ws, n, stage_half, true);
    }

    if step + 1 == total_steps {
        if state.is_zero {
            for k in 0..half {
                emit(k, 0);
            }
            return Ok(0);
        }
        let mut sat_count = 0u64;
        let shift = state.exp_acc + frac_bits as i32;
        for k in 0..half {
            let v = load_i32_from_i16(re_ws, (half + k) * 2);
            let scaled = scale_pow2(v as i64, shift);
            let out = if scaled > i32::MAX as i64 {
                sat_count = sat_count.saturating_add(1);
                i32::MAX
            } else if scaled < i32::MIN as i64 {
                sat_count = sat_count.saturating_add(1);
                i32::MIN
            } else {
                scaled as i32
            };
            emit(k, out);
        }
        Ok(sat_count)
    } else {
        Ok(0)
    }
}
