//! Single quarter-wave `f64`-derived IEEE 754 RNE twiddle ROM for `SimdFixedFftBackend`.
//!
//! Stores `TW_COS_Q[k] = cos(2 * pi * k / 16384)` for `k = 0..=4096` (`4097` `f32` values =
//! `16,388 B` ROM), covering every FFT size `N = 64 ..= 16384` via power-of-two striding.
//! Cardinal points are exact (`TW_COS_Q[0] = 1.0`, `TW_COS_Q[4096] = 0.0`).
//! Bit reversal is computed on the fly via `usize::reverse_bits()`, using `0 B` of ROM.

include!(concat!(env!("OUT_DIR"), "/simd_fft_tables_gen.rs"));

/// Exact ROM size in bytes of the `SimdFixedFftBackend` twiddle table (`16,388 B`).
pub(crate) const SIMD_FFT_ROM_BYTES: usize = (TW_QUARTER + 1) * core::mem::size_of::<f32>();

/// First-quadrant `(cos, sin)` lookup for `idx in 0..=4096` (`[0, pi/2]`), branchless.
#[inline(always)]
pub(crate) fn twiddle_q1(idx: usize) -> (f32, f32) {
    (TW_COS_Q[idx], TW_COS_Q[TW_QUARTER - idx])
}
