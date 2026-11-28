//! Frequency-domain pipeline shared by the fixed-point partitioned engines, built on the
//! 32-bit-mantissa BFP real FFT core ([`crate::fixed::fft_q31`]).
//!
//! Storage stays Q1.15 block-floating-point (`i16` mantissas plus one exponent per partition
//! spectrum, convention `X(k) ≈ mant·2^(exp − 15)`), so the workspace budget of the Q1.15
//! engines is unchanged; every transform runs with ~29-bit mantissas and every spectrum is
//! rounded to 16 bits exactly once (round to nearest even).
//!
//! Per block: forward real FFT of the overlap-save window → Q1.15 BFP slot; exponent-aligned
//! complex multiply-accumulate of all partitions into exact `i64` accumulators with guard bits
//! ([`cmac_guard_bits`]); in-place normalization of the accumulators to `i32` mantissas
//! ([`crate::fixed::fft_q31::acc_i64_to_q31_spectrum_in_place`]) and inverse real FFT inside
//! the same accumulator region. No intermediate 16-bit requantization of the result spectrum.

use crate::fixed::block_float::{
    TapSource, load_i64_from_i16, round_shift_even_i64, store_i64_to_i16,
};
use crate::fixed::fft_q31::{
    Q31FftState, ZERO_SPECTRUM_EXP, q31_fft_steps, q31_rfft_step, q31_spectrum_to_q15,
};

/// `i16` words of the forward transform buffer for FFT length `n` (`2 · (N + 2)`).
#[inline]
pub(crate) const fn rfft_buf_i16(n: usize) -> usize {
    2 * crate::fixed::fft_q31::q31_rfft_words(n)
}

/// `i16` words that the in-place inverse transform needs inside an accumulator region
/// (`2 · (2N + 2) = 4N + 4 <= 8 · bins`).
#[inline]
pub(crate) const fn irfft_buf_i16(n: usize) -> usize {
    2 * crate::fixed::fft_q31::q31_irfft_words(n)
}

/// Stores an exponent in an `i16` workspace word (saturating; real exponents are tiny).
#[inline(always)]
pub(crate) fn exp_to_word(e: i32) -> i16 {
    e.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Guard bits for the exponent-aligned `i64` accumulation of `partitions` products: each
/// product `|xr·hr − xi·hi| < 2^31` is scaled by up to `2^G`, so the sum stays below
/// `partitions · 2^(31+G) <= 2^62` with `G = 31 − bitlen(partitions)`. Products whose exponent
/// is within `G` of the block maximum are accumulated exactly.
#[inline]
pub(crate) const fn cmac_guard_bits(partitions: usize) -> u32 {
    let bits = usize::BITS - partitions.leading_zeros();
    let g = 31i32 - bits as i32;
    if g < 0 {
        0
    } else if g > 30 {
        30
    } else {
        g as u32
    }
}

/// Forward-transforms taps `start..start + len` of `ir` (zero-padded to `n`) as Q1.31 and
/// stores the spectrum as Q1.15 BFP in `re`/`im`, returning the stored exponent.
pub(crate) fn prepare_partition<S: TapSource>(
    ir: &S,
    start: usize,
    len: usize,
    n: usize,
    buf: &mut [i16],
    re: &mut [i16],
    im: &mut [i16],
) -> i32 {
    let mut st = Q31FftState::new();
    let tap = |i: usize| if i < len { ir.tap_q31(start + i) } else { 0 };
    let mut e = ZERO_SPECTRUM_EXP;
    for step in 0..q31_fft_steps(n) {
        if let Some(x) = q31_rfft_step(n, step, tap, -31, buf, &mut st) {
            e = x;
        }
    }
    q31_spectrum_to_q15(buf, n, e, re, im)
}

/// Accumulates one partition product `X·H` into the packed `i64` accumulators at `acc`
/// (`acc_re` words `0..4·bins`, `acc_im` words `4·bins..8·bins`), scaled by `2^sh`
/// (`sh = G − (max_exp − pe)`: exact left shift for `sh >= 0`, round-to-nearest-even right
/// shift otherwise; products more than `62` bits below are dropped).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cmac_partition(
    ws: &mut [i16],
    acc: usize,
    bins: usize,
    x_re: usize,
    x_im: usize,
    h_re: usize,
    h_im: usize,
    sh: i32,
) {
    if sh < -62 {
        return;
    }
    let acc_im = acc + 4 * bins;
    for k in 0..bins {
        let xr = ws[x_re + k] as i64;
        let xi = ws[x_im + k] as i64;
        let hr = ws[h_re + k] as i64;
        let hi = ws[h_im + k] as i64;
        let (pr, pi) = (xr * hr - xi * hi, xr * hi + xi * hr);
        let (pr, pi) = if sh >= 0 {
            (pr << sh, pi << sh)
        } else {
            let s = sh.unsigned_abs();
            (round_shift_even_i64(pr, s), round_shift_even_i64(pi, s))
        };
        let ro = acc + 4 * k;
        let io = acc_im + 4 * k;
        store_i64_to_i16(ws, ro, load_i64_from_i16(ws, ro) + pr);
        store_i64_to_i16(ws, io, load_i64_from_i16(ws, io) + pi);
    }
}
