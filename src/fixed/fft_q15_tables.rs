//! Pre-computed Q1.15 quarter-wave twiddle ROM for `N = 64 ..= 2048`.
//!
//! Stores `TWIDDLE_COS_Q_2048[k] = round_ties_even(32768 * cos(2 * pi * k / 2048))` clamped to
//! `i16` for `k = 0..=512` (`513` entries = `1,026 B` ROM), generated in `build.rs`.
//! Exact unit values (`cos(0) = 1`, `sin(pi/2) = 1`) are returned as `32768` in `i32` by
//! `fft_q15::twiddle_q15`.

include!(concat!(env!("OUT_DIR"), "/fft_q15_tables_gen.rs"));

/// Exact ROM size in bytes of the Q1.15 FFT twiddle table (`1,026 B`).
pub(crate) const FIXED_FFT_ROM_BYTES: usize = (Q15_TW_QUARTER + 1) * core::mem::size_of::<i16>();
