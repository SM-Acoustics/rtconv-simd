//! Build-time generated Q1.31 quarter-wave twiddle ROM (`2,052 B`) for the 32-bit-mantissa
//! block-floating-point FFT (`N <= 2048`); see `build.rs`.

include!(concat!(env!("OUT_DIR"), "/fft_q31_tables_gen.rs"));
