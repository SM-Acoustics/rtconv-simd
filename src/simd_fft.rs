//! Embedded `#![no_std]`, zero-allocation SIMD FFT backend (`SimdFixedFftBackend`)
//! for `N = 64 ..= 16384` using `fearless_simd 1.2.0`.
//!
//! ## Design
//! - Computes `N`-point RFFT and `N`-point normalized C2R IFFT via an inner
//!   `M = N / 2`-point split-complex (SoA) DIT FFT (`M = 32 ..= 8192`), halving
//!   the complex transform work compared to a full `N`-point complex IFFT.
//! - Uses caller-provided external scratch (`N` floats = `re[0..M]` + `im[0..M]`),
//!   avoiding large stack allocations on microcontrollers.
//! - Single precomputed quarter-wave cosine ROM (`16,388 B` in `.rodata` / Flash,
//!   derived in `f64` and rounded to `f32` under IEEE 754 RNE via `build.rs`),
//!   indexed with power-of-two stride across all sizes `N = 64 ..= 16384`.
//! - Bit-reversal indices are computed via `usize::reverse_bits()` (`0 B` bitrev ROM).
//! - Stages 1 & 2 (`half = 1, 2`) are fused into a multiplication-free radix-4 pass.
//! - Stages `half = 4 ..= M/2` use explicit `fearless_simd 1.2.0` vector butterflies
//!   (`f32x4` for `half = 4` and native `S::f32s` for `half >= LEN`).
//! - Falls back cleanly to scalar butterflies when `SimdBackend::scalar()` is
//!   selected or when the target has no SIMD support (e.g. RISC-V).

use crate::SimdBackend;
use crate::error::ConvError;
use crate::fft::{FftBackend, FftSize};
use crate::simd_fft_tables::{MAX_FFT_SIZE, SIMD_FFT_ROM_BYTES, TW_QUARTER, twiddle_q1};
use fearless_simd::{Simd, dispatch, f32x4, prelude::*};
use fearless_simd_macros::simd;

/// Allocation-free, `#![no_std]` SIMD FFT backend supporting `N = 64 ..= 16384`.
#[derive(Clone, Copy, Debug)]
pub struct SimdFixedFftBackend {
    backend: SimdBackend,
}

impl SimdFixedFftBackend {
    /// Creates a new `SimdFixedFftBackend` bound to a pre-validated [`SimdBackend`].
    #[inline]
    pub const fn new(backend: SimdBackend) -> Self {
        Self { backend }
    }

    /// Returns the configured [`SimdBackend`].
    #[inline]
    pub const fn simd_backend(&self) -> SimdBackend {
        self.backend
    }

    /// Exact read-only twiddle table size in bytes (`16,388 B`, single quarter-wave table
    /// shared across `N64` through `N16384`; `0 B` bit-reversal ROM).
    #[inline]
    pub const fn rom_table_bytes() -> usize {
        SIMD_FFT_ROM_BYTES
    }
}

/// Fused radix-4 pass covering stages `half = 1` and `half = 2` (multiplication-free).
#[inline(always)]
fn radix4_stage01(re: &mut [f32], im: &mut [f32], m: usize, inverse: bool) {
    let mut i = 0;
    if !inverse {
        while i + 4 <= m {
            let r0 = re[i];
            let i0 = im[i];
            let r1 = re[i + 1];
            let i1 = im[i + 1];
            let r2 = re[i + 2];
            let i2 = im[i + 2];
            let r3 = re[i + 3];
            let i3 = im[i + 3];

            let a0r = r0 + r1;
            let a0i = i0 + i1;
            let a1r = r0 - r1;
            let a1i = i0 - i1;
            let a2r = r2 + r3;
            let a2i = i2 + i3;
            let a3r = r2 - r3;
            let a3i = i2 - i3;

            // Forward W_4^1 = -j: (-j) * (a3r + j*a3i) = a3i - j*a3r
            re[i] = a0r + a2r;
            im[i] = a0i + a2i;
            re[i + 2] = a0r - a2r;
            im[i + 2] = a0i - a2i;
            re[i + 1] = a1r + a3i;
            im[i + 1] = a1i - a3r;
            re[i + 3] = a1r - a3i;
            im[i + 3] = a1i + a3r;

            i += 4;
        }
    } else {
        while i + 4 <= m {
            let r0 = re[i];
            let i0 = im[i];
            let r1 = re[i + 1];
            let i1 = im[i + 1];
            let r2 = re[i + 2];
            let i2 = im[i + 2];
            let r3 = re[i + 3];
            let i3 = im[i + 3];

            let a0r = r0 + r1;
            let a0i = i0 + i1;
            let a1r = r0 - r1;
            let a1i = i0 - i1;
            let a2r = r2 + r3;
            let a2i = i2 + i3;
            let a3r = r2 - r3;
            let a3i = i2 - i3;

            // Inverse W_4^{-1} = +j: (+j) * (a3r + j*a3i) = -a3i + j*a3r
            re[i] = a0r + a2r;
            im[i] = a0i + a2i;
            re[i + 2] = a0r - a2r;
            im[i + 2] = a0i - a2i;
            re[i + 1] = a1r - a3i;
            im[i + 1] = a1i + a3r;
            re[i + 3] = a1r + a3i;
            im[i + 3] = a1i - a3r;

            i += 4;
        }
    }
}

/// Scalar butterfly stage for any `half >= 4` using the strided quarter-wave ROM.
/// Keeps `base` as the outer loop for sequential L1d cache locality and splits `j` at `half / 2`
/// into quadrant 1 and quadrant 2 to stay branchless in the inner loop.
#[inline(always)]
fn stage_scalar(re: &mut [f32], im: &mut [f32], m: usize, half: usize, inverse: bool) {
    let step = half * 2;
    let q_half = half / 2;
    let tw_stride = (MAX_FFT_SIZE / 2) / half;
    let mut base = 0;
    if !inverse {
        while base < m {
            let mut j = 0;
            let mut idx = 0;
            while j < q_half {
                let (wr, wi) = twiddle_q1(idx);
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = re[i0];
                let ui = im[i0];
                let vr = re[i1];
                let vi = im[i1];
                let tr = vr * wr + vi * wi;
                let ti = vi * wr - vr * wi;
                re[i0] = ur + tr;
                im[i0] = ui + ti;
                re[i1] = ur - tr;
                im[i1] = ui - ti;
                idx += tw_stride;
                j += 1;
            }
            let mut rem = 0;
            while j < half {
                let (s, c_neg) = twiddle_q1(rem);
                let wr = -c_neg;
                let wi = s;
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = re[i0];
                let ui = im[i0];
                let vr = re[i1];
                let vi = im[i1];
                let tr = vr * wr + vi * wi;
                let ti = vi * wr - vr * wi;
                re[i0] = ur + tr;
                im[i0] = ui + ti;
                re[i1] = ur - tr;
                im[i1] = ui - ti;
                rem += tw_stride;
                j += 1;
            }
            base += step;
        }
    } else {
        while base < m {
            let mut j = 0;
            let mut idx = 0;
            while j < q_half {
                let (wr, wi) = twiddle_q1(idx);
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = re[i0];
                let ui = im[i0];
                let vr = re[i1];
                let vi = im[i1];
                let tr = vr * wr - vi * wi;
                let ti = vr * wi + vi * wr;
                re[i0] = ur + tr;
                im[i0] = ui + ti;
                re[i1] = ur - tr;
                im[i1] = ui - ti;
                idx += tw_stride;
                j += 1;
            }
            let mut rem = 0;
            while j < half {
                let (s, c_neg) = twiddle_q1(rem);
                let wr = -c_neg;
                let wi = s;
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = re[i0];
                let ui = im[i0];
                let vr = re[i1];
                let vi = im[i1];
                let tr = vr * wr - vi * wi;
                let ti = vr * wi + vi * wr;
                re[i0] = ur + tr;
                im[i0] = ui + ti;
                re[i1] = ur - tr;
                im[i1] = ui - ti;
                rem += tw_stride;
                j += 1;
            }
            base += step;
        }
    }
}

/// 4-wide SIMD butterfly stage (`half == 4`), works cleanly on SSE2, AVX2, NEON, WASM.
#[inline(always)]
fn stage_half4_simd<S: Simd>(simd: S, re: &mut [f32], im: &mut [f32], m: usize, inverse: bool) {
    let (c1, s1) = twiddle_q1(TW_QUARTER / 2); // cos(pi/4), sin(pi/4)
    let tw_cos = [1.0f32, c1, 0.0f32, -c1];
    let tw_sin = [0.0f32, s1, 1.0f32, s1];
    let wr = f32x4::from_slice(simd, &tw_cos);
    let wi = f32x4::from_slice(simd, &tw_sin);
    let mut base = 0;
    if !inverse {
        while base + 8 <= m {
            let ur = f32x4::from_slice(simd, &re[base..base + 4]);
            let ui = f32x4::from_slice(simd, &im[base..base + 4]);
            let vr = f32x4::from_slice(simd, &re[base + 4..base + 8]);
            let vi = f32x4::from_slice(simd, &im[base + 4..base + 8]);

            let tr = vr.mul_add(wr, vi * wi);
            let ti = vi.mul_add(wr, -(vr * wi));

            (ur + tr).store_slice(&mut re[base..base + 4]);
            (ui + ti).store_slice(&mut im[base..base + 4]);
            (ur - tr).store_slice(&mut re[base + 4..base + 8]);
            (ui - ti).store_slice(&mut im[base + 4..base + 8]);
            base += 8;
        }
    } else {
        while base + 8 <= m {
            let ur = f32x4::from_slice(simd, &re[base..base + 4]);
            let ui = f32x4::from_slice(simd, &im[base..base + 4]);
            let vr = f32x4::from_slice(simd, &re[base + 4..base + 8]);
            let vi = f32x4::from_slice(simd, &im[base + 4..base + 8]);

            let tr = vr.mul_add(wr, -(vi * wi));
            let ti = vr.mul_add(wi, vi * wr);

            (ur + tr).store_slice(&mut re[base..base + 4]);
            (ui + ti).store_slice(&mut im[base..base + 4]);
            (ur - tr).store_slice(&mut re[base + 4..base + 8]);
            (ui - ti).store_slice(&mut im[base + 4..base + 8]);
            base += 8;
        }
    }
}

/// Register-hoisted SIMD butterfly stage for `half == 8`: loads all 8 twiddles once outside the
/// sequential `base` loop (1 native vector on AVX2, 2 `f32x4` vectors on SSE2/NEON/WASM).
#[inline(always)]
fn stage_half8_simd<S: Simd>(simd: S, re: &mut [f32], im: &mut [f32], m: usize, inverse: bool) {
    let q = TW_QUARTER / 4;
    let (c1, s1) = twiddle_q1(q);
    let (c2, s2) = twiddle_q1(2 * q);
    let (c3, s3) = twiddle_q1(3 * q);
    let tw_cos = [1.0f32, c1, c2, c3, 0.0f32, -c3, -c2, -c1];
    let tw_sin = [0.0f32, s1, s2, s3, 1.0f32, s3, s2, s1];

    if S::f32s::LEN == 8 {
        let wr = S::f32s::from_slice(simd, &tw_cos);
        let wi = S::f32s::from_slice(simd, &tw_sin);
        let mut base = 0;
        if !inverse {
            while base + 16 <= m {
                let ur = S::f32s::from_slice(simd, &re[base..base + 8]);
                let ui = S::f32s::from_slice(simd, &im[base..base + 8]);
                let vr = S::f32s::from_slice(simd, &re[base + 8..base + 16]);
                let vi = S::f32s::from_slice(simd, &im[base + 8..base + 16]);

                let tr = vr.mul_add(wr, vi * wi);
                let ti = vi.mul_add(wr, -(vr * wi));

                (ur + tr).store_slice(&mut re[base..base + 8]);
                (ui + ti).store_slice(&mut im[base..base + 8]);
                (ur - tr).store_slice(&mut re[base + 8..base + 16]);
                (ui - ti).store_slice(&mut im[base + 8..base + 16]);
                base += 16;
            }
        } else {
            while base + 16 <= m {
                let ur = S::f32s::from_slice(simd, &re[base..base + 8]);
                let ui = S::f32s::from_slice(simd, &im[base..base + 8]);
                let vr = S::f32s::from_slice(simd, &re[base + 8..base + 16]);
                let vi = S::f32s::from_slice(simd, &im[base + 8..base + 16]);

                let tr = vr.mul_add(wr, -(vi * wi));
                let ti = vr.mul_add(wi, vi * wr);

                (ur + tr).store_slice(&mut re[base..base + 8]);
                (ui + ti).store_slice(&mut im[base..base + 8]);
                (ur - tr).store_slice(&mut re[base + 8..base + 16]);
                (ui - ti).store_slice(&mut im[base + 8..base + 16]);
                base += 16;
            }
        }
        return;
    }

    let wr0 = f32x4::from_slice(simd, &tw_cos[0..4]);
    let wi0 = f32x4::from_slice(simd, &tw_sin[0..4]);
    let wr1 = f32x4::from_slice(simd, &tw_cos[4..8]);
    let wi1 = f32x4::from_slice(simd, &tw_sin[4..8]);
    let mut base = 0;
    if !inverse {
        while base + 16 <= m {
            let ur0 = f32x4::from_slice(simd, &re[base..base + 4]);
            let ui0 = f32x4::from_slice(simd, &im[base..base + 4]);
            let vr0 = f32x4::from_slice(simd, &re[base + 8..base + 12]);
            let vi0 = f32x4::from_slice(simd, &im[base + 8..base + 12]);
            let tr0 = vr0.mul_add(wr0, vi0 * wi0);
            let ti0 = vi0.mul_add(wr0, -(vr0 * wi0));
            (ur0 + tr0).store_slice(&mut re[base..base + 4]);
            (ui0 + ti0).store_slice(&mut im[base..base + 4]);
            (ur0 - tr0).store_slice(&mut re[base + 8..base + 12]);
            (ui0 - ti0).store_slice(&mut im[base + 8..base + 12]);

            let ur1 = f32x4::from_slice(simd, &re[base + 4..base + 8]);
            let ui1 = f32x4::from_slice(simd, &im[base + 4..base + 8]);
            let vr1 = f32x4::from_slice(simd, &re[base + 12..base + 16]);
            let vi1 = f32x4::from_slice(simd, &im[base + 12..base + 16]);
            let tr1 = vr1.mul_add(wr1, vi1 * wi1);
            let ti1 = vi1.mul_add(wr1, -(vr1 * wi1));
            (ur1 + tr1).store_slice(&mut re[base + 4..base + 8]);
            (ui1 + ti1).store_slice(&mut im[base + 4..base + 8]);
            (ur1 - tr1).store_slice(&mut re[base + 12..base + 16]);
            (ui1 - ti1).store_slice(&mut im[base + 12..base + 16]);
            base += 16;
        }
    } else {
        while base + 16 <= m {
            let ur0 = f32x4::from_slice(simd, &re[base..base + 4]);
            let ui0 = f32x4::from_slice(simd, &im[base..base + 4]);
            let vr0 = f32x4::from_slice(simd, &re[base + 8..base + 12]);
            let vi0 = f32x4::from_slice(simd, &im[base + 8..base + 12]);
            let tr0 = vr0.mul_add(wr0, -(vi0 * wi0));
            let ti0 = vr0.mul_add(wi0, vi0 * wr0);
            (ur0 + tr0).store_slice(&mut re[base..base + 4]);
            (ui0 + ti0).store_slice(&mut im[base..base + 4]);
            (ur0 - tr0).store_slice(&mut re[base + 8..base + 12]);
            (ui0 - ti0).store_slice(&mut im[base + 8..base + 12]);

            let ur1 = f32x4::from_slice(simd, &re[base + 4..base + 8]);
            let ui1 = f32x4::from_slice(simd, &im[base + 4..base + 8]);
            let vr1 = f32x4::from_slice(simd, &re[base + 12..base + 16]);
            let vi1 = f32x4::from_slice(simd, &im[base + 12..base + 16]);
            let tr1 = vr1.mul_add(wr1, -(vi1 * wi1));
            let ti1 = vr1.mul_add(wi1, vi1 * wr1);
            (ur1 + tr1).store_slice(&mut re[base + 4..base + 8]);
            (ui1 + ti1).store_slice(&mut im[base + 4..base + 8]);
            (ur1 - tr1).store_slice(&mut re[base + 12..base + 16]);
            (ui1 - ti1).store_slice(&mut im[base + 12..base + 16]);
            base += 16;
        }
    }
}

/// Loads `lanes` quadrant-1 twiddles starting at `idx0` (`idx_last <= TW_QUARTER`).
#[inline(always)]
fn load_twiddle_q1_vec<S: Simd>(simd: S, idx0: usize, tw_stride: usize) -> (S::f32s, S::f32s) {
    let lanes = S::f32s::LEN;
    let mut wr_buf = [0.0f32; 16];
    let mut wi_buf = [0.0f32; 16];
    let mut l = 0;
    let mut idx = idx0;
    while l < lanes {
        let (c, s) = twiddle_q1(idx);
        wr_buf[l] = c;
        wi_buf[l] = s;
        idx += tw_stride;
        l += 1;
    }
    (
        S::f32s::from_slice(simd, &wr_buf[..lanes]),
        S::f32s::from_slice(simd, &wi_buf[..lanes]),
    )
}

/// Loads `lanes` quadrant-2 twiddles starting at `rem0 = idx0 - TW_QUARTER`.
#[inline(always)]
fn load_twiddle_q2_vec<S: Simd>(simd: S, rem0: usize, tw_stride: usize) -> (S::f32s, S::f32s) {
    let lanes = S::f32s::LEN;
    let mut wr_buf = [0.0f32; 16];
    let mut wi_buf = [0.0f32; 16];
    let mut l = 0;
    let mut rem = rem0;
    while l < lanes {
        let (s, c_neg) = twiddle_q1(rem);
        wr_buf[l] = -c_neg;
        wi_buf[l] = s;
        rem += tw_stride;
        l += 1;
    }
    (
        S::f32s::from_slice(simd, &wr_buf[..lanes]),
        S::f32s::from_slice(simd, &wi_buf[..lanes]),
    )
}

/// Native-width SIMD butterfly stage for any `half >= 16` (branchless Q1/Q2 split on `j`,
/// hoisting each twiddle vector load outside the `base` group loop).
#[inline(always)]
fn stage_native_simd<S: Simd>(
    simd: S,
    re: &mut [f32],
    im: &mut [f32],
    m: usize,
    half: usize,
    inverse: bool,
) {
    let lanes = S::f32s::LEN;
    let q_half = half / 2;
    if q_half < lanes || !q_half.is_multiple_of(lanes) || lanes > 16 {
        stage_scalar(re, im, m, half, inverse);
        return;
    }
    let step = half * 2;
    let tw_stride = (MAX_FFT_SIZE / 2) / half;
    let vec_tw_step = lanes * tw_stride;

    let mut j = 0;
    let mut idx = 0;
    if !inverse {
        while j < q_half {
            let (wr, wi) = load_twiddle_q1_vec(simd, idx, tw_stride);
            let mut base = 0;
            while base < m {
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = S::f32s::from_slice(simd, &re[i0..i0 + lanes]);
                let ui = S::f32s::from_slice(simd, &im[i0..i0 + lanes]);
                let vr = S::f32s::from_slice(simd, &re[i1..i1 + lanes]);
                let vi = S::f32s::from_slice(simd, &im[i1..i1 + lanes]);

                let tr = vr.mul_add(wr, vi * wi);
                let ti = vi.mul_add(wr, -(vr * wi));

                (ur + tr).store_slice(&mut re[i0..i0 + lanes]);
                (ui + ti).store_slice(&mut im[i0..i0 + lanes]);
                (ur - tr).store_slice(&mut re[i1..i1 + lanes]);
                (ui - ti).store_slice(&mut im[i1..i1 + lanes]);
                base += step;
            }
            idx += vec_tw_step;
            j += lanes;
        }
        let mut rem = 0;
        while j < half {
            let (wr, wi) = load_twiddle_q2_vec(simd, rem, tw_stride);
            let mut base = 0;
            while base < m {
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = S::f32s::from_slice(simd, &re[i0..i0 + lanes]);
                let ui = S::f32s::from_slice(simd, &im[i0..i0 + lanes]);
                let vr = S::f32s::from_slice(simd, &re[i1..i1 + lanes]);
                let vi = S::f32s::from_slice(simd, &im[i1..i1 + lanes]);

                let tr = vr.mul_add(wr, vi * wi);
                let ti = vi.mul_add(wr, -(vr * wi));

                (ur + tr).store_slice(&mut re[i0..i0 + lanes]);
                (ui + ti).store_slice(&mut im[i0..i0 + lanes]);
                (ur - tr).store_slice(&mut re[i1..i1 + lanes]);
                (ui - ti).store_slice(&mut im[i1..i1 + lanes]);
                base += step;
            }
            rem += vec_tw_step;
            j += lanes;
        }
    } else {
        while j < q_half {
            let (wr, wi) = load_twiddle_q1_vec(simd, idx, tw_stride);
            let mut base = 0;
            while base < m {
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = S::f32s::from_slice(simd, &re[i0..i0 + lanes]);
                let ui = S::f32s::from_slice(simd, &im[i0..i0 + lanes]);
                let vr = S::f32s::from_slice(simd, &re[i1..i1 + lanes]);
                let vi = S::f32s::from_slice(simd, &im[i1..i1 + lanes]);

                let tr = vr.mul_add(wr, -(vi * wi));
                let ti = vr.mul_add(wi, vi * wr);

                (ur + tr).store_slice(&mut re[i0..i0 + lanes]);
                (ui + ti).store_slice(&mut im[i0..i0 + lanes]);
                (ur - tr).store_slice(&mut re[i1..i1 + lanes]);
                (ui - ti).store_slice(&mut im[i1..i1 + lanes]);
                base += step;
            }
            idx += vec_tw_step;
            j += lanes;
        }
        let mut rem = 0;
        while j < half {
            let (wr, wi) = load_twiddle_q2_vec(simd, rem, tw_stride);
            let mut base = 0;
            while base < m {
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = S::f32s::from_slice(simd, &re[i0..i0 + lanes]);
                let ui = S::f32s::from_slice(simd, &im[i0..i0 + lanes]);
                let vr = S::f32s::from_slice(simd, &re[i1..i1 + lanes]);
                let vi = S::f32s::from_slice(simd, &im[i1..i1 + lanes]);

                let tr = vr.mul_add(wr, -(vi * wi));
                let ti = vr.mul_add(wi, vi * wr);

                (ur + tr).store_slice(&mut re[i0..i0 + lanes]);
                (ui + ti).store_slice(&mut im[i0..i0 + lanes]);
                (ur - tr).store_slice(&mut re[i1..i1 + lanes]);
                (ui - ti).store_slice(&mut im[i1..i1 + lanes]);
                base += step;
            }
            rem += vec_tw_step;
            j += lanes;
        }
    }
}

#[inline(always)]
fn cfft_core_scalar(re: &mut [f32], im: &mut [f32], m: usize, inverse: bool) {
    radix4_stage01(re, im, m, inverse);
    let mut half = 4;
    while half < m {
        stage_scalar(re, im, m, half, inverse);
        half <<= 1;
    }
}

#[inline(always)]
fn cfft_core_simd<S: Simd>(simd: S, re: &mut [f32], im: &mut [f32], m: usize, inverse: bool) {
    radix4_stage01(re, im, m, inverse);
    stage_half4_simd(simd, re, im, m, inverse);
    if m > 8 {
        stage_half8_simd(simd, re, im, m, inverse);
    }
    let mut half = 16;
    while half < m {
        stage_native_simd(simd, re, im, m, half, inverse);
        half <<= 1;
    }
}

#[inline(always)]
fn r2c_untangle(z_re: &[f32], z_im: &[f32], spec_re: &mut [f32], spec_im: &mut [f32], m: usize) {
    // k = 0 (DC) and k = M (Nyquist)
    spec_re[0] = z_re[0] + z_im[0];
    spec_im[0] = 0.0;
    spec_re[m] = z_re[0] - z_im[0];
    spec_im[m] = 0.0;

    // k = M/2
    let half_m = m / 2;
    spec_re[half_m] = z_re[half_m];
    spec_im[half_m] = -z_im[half_m];

    // Stride into the N_MAX = 16384 quarter-wave table for N = 2*M:
    // angle = 2*pi*k / (2*M) = 2*pi*(k * (MAX_FFT_SIZE / (2*M))) / MAX_FFT_SIZE.
    let r2c_stride = (MAX_FFT_SIZE / 2) / m;
    let mut k = 1;
    while k < half_m {
        let mk = m - k;
        let a = z_re[k];
        let b = z_im[k];
        let c = z_re[mk];
        let d = z_im[mk];

        let a_re = 0.5 * (a + c);
        let a_im = 0.5 * (b - d);
        let b_re = 0.5 * (b + d);
        let b_im = 0.5 * (c - a);

        let (wc, ws) = twiddle_q1(k * r2c_stride);

        // W_N^k * B = (wc - j*ws) * (b_re + j*b_im)
        let wb_re = wc * b_re + ws * b_im;
        let wb_im = wc * b_im - ws * b_re;

        spec_re[k] = a_re + wb_re;
        spec_im[k] = a_im + wb_im;
        spec_re[mk] = a_re - wb_re;
        spec_im[mk] = -a_im + wb_im;
        k += 1;
    }
}

#[inline(always)]
fn c2r_preprocess_bitrev(
    spec_re: &[f32],
    spec_im: &[f32],
    re: &mut [f32],
    im: &mut [f32],
    m: usize,
) {
    // Fold 1/N = 0.5/M directly into the pre-process pass.
    let inv_n = 0.5 / (m as f32);
    let inv_m = 1.0 / (m as f32);
    let bitrev_shift = usize::BITS - m.trailing_zeros();

    // k = 0 -> bitrev(0) = 0
    re[0] = inv_n * (spec_re[0] + spec_re[m]);
    im[0] = inv_n * (spec_re[0] - spec_re[m]);

    // k = M/2
    let half_m = m / 2;
    let br_mid = half_m.reverse_bits() >> bitrev_shift;
    re[br_mid] = inv_m * spec_re[half_m];
    im[br_mid] = -inv_m * spec_im[half_m];

    let r2c_stride = (MAX_FFT_SIZE / 2) / m;
    let mut k = 1;
    while k < half_m {
        let mk = m - k;
        let xr1 = spec_re[k];
        let xi1 = spec_im[k];
        let xr2 = spec_re[mk];
        let xi2 = spec_im[mk];

        let e_re = inv_n * (xr1 + xr2);
        let e_im = inv_n * (xi1 - xi2);
        let d_re = inv_n * (xr1 - xr2);
        let d_im = inv_n * (xi1 + xi2);

        let (wc, ws) = twiddle_q1(k * r2c_stride);

        // conj(W_N^k) * D = (wc + j*ws) * (d_re + j*d_im)
        let o_re = wc * d_re - ws * d_im;
        let o_im = wc * d_im + ws * d_re;

        let br_k = k.reverse_bits() >> bitrev_shift;
        let br_mk = mk.reverse_bits() >> bitrev_shift;
        re[br_k] = e_re - o_im;
        im[br_k] = e_im + o_re;
        re[br_mk] = e_re + o_im;
        im[br_mk] = -e_im + o_re;
        k += 1;
    }
}

#[inline(always)]
fn pack_time_bitrev(time_in: &[f32], re: &mut [f32], im: &mut [f32], m: usize) {
    let bitrev_shift = usize::BITS - m.trailing_zeros();
    let mut k = 0;
    while k < m {
        let br = k.reverse_bits() >> bitrev_shift;
        re[k] = time_in[2 * br];
        im[k] = time_in[2 * br + 1];
        k += 1;
    }
}

#[simd]
fn rfft_forward_simd_impl<S: Simd>(
    simd: S,
    size: FftSize,
    time_in: &[f32],
    spec_re: &mut [f32],
    spec_im: &mut [f32],
    scratch: &mut [f32],
) {
    let m = size.half_len();
    let (re, im) = scratch[..2 * m].split_at_mut(m);
    pack_time_bitrev(time_in, re, im, m);
    cfft_core_simd(simd, re, im, m, false);
    r2c_untangle(re, im, spec_re, spec_im, m);
}

#[simd]
fn rfft_inverse_simd_impl<S: Simd>(
    simd: S,
    size: FftSize,
    spec_re: &[f32],
    spec_im: &[f32],
    time_out: &mut [f32],
    scratch: &mut [f32],
) {
    let m = size.half_len();
    let (re, im) = scratch[..2 * m].split_at_mut(m);
    c2r_preprocess_bitrev(spec_re, spec_im, re, im, m);
    cfft_core_simd(simd, re, im, m, true);
    let mut k = 0;
    while k < m {
        time_out[2 * k] = re[k];
        time_out[2 * k + 1] = im[k];
        k += 1;
    }
}

impl FftBackend for SimdFixedFftBackend {
    #[inline]
    fn name(&self) -> &'static str {
        if self.backend.is_scalar() || self.backend.level().is_fallback() {
            "simd_fixed_fft (no_std scalar)"
        } else {
            "simd_fixed_fft (no_std fearless_simd 1.2)"
        }
    }

    #[inline]
    fn is_simd_accelerated(&self) -> bool {
        !self.backend.is_scalar() && !self.backend.level().is_fallback()
    }

    #[inline]
    fn supports_size(&self, _size: FftSize) -> bool {
        true
    }

    #[inline]
    fn scratch_len_f32(&self, size: FftSize) -> usize {
        // External scratch: M floats for re + M floats for im = N floats.
        size.len()
    }

    fn rfft_forward(
        &self,
        size: FftSize,
        time_in: &[f32],
        spec_re: &mut [f32],
        spec_im: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), ConvError> {
        let n = size.len();
        let bins = size.spectrum_bins();
        if time_in.len() != n {
            return Err(ConvError::MismatchedBufferLength {
                expected: n,
                actual: time_in.len(),
            });
        }
        if spec_re.len() < bins || spec_im.len() < bins {
            return Err(ConvError::MismatchedBufferLength {
                expected: bins,
                actual: spec_re.len().min(spec_im.len()),
            });
        }
        if scratch.len() < n {
            return Err(ConvError::WorkspaceTooSmall {
                required: n,
                provided: scratch.len(),
            });
        }

        if self.backend.is_scalar() || self.backend.level().is_fallback() {
            let m = size.half_len();
            let (re, im) = scratch[..2 * m].split_at_mut(m);
            pack_time_bitrev(time_in, re, im, m);
            cfft_core_scalar(re, im, m, false);
            r2c_untangle(re, im, spec_re, spec_im, m);
        } else {
            dispatch!(
                self.backend.level(),
                simd => rfft_forward_simd_impl(simd, size, time_in, spec_re, spec_im, scratch)
            );
        }
        Ok(())
    }

    fn rfft_inverse(
        &self,
        size: FftSize,
        spec_re: &[f32],
        spec_im: &[f32],
        time_out: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), ConvError> {
        let n = size.len();
        let bins = size.spectrum_bins();
        if time_out.len() != n {
            return Err(ConvError::MismatchedBufferLength {
                expected: n,
                actual: time_out.len(),
            });
        }
        if spec_re.len() < bins || spec_im.len() < bins {
            return Err(ConvError::MismatchedBufferLength {
                expected: bins,
                actual: spec_re.len().min(spec_im.len()),
            });
        }
        if scratch.len() < n {
            return Err(ConvError::WorkspaceTooSmall {
                required: n,
                provided: scratch.len(),
            });
        }

        if self.backend.is_scalar() || self.backend.level().is_fallback() {
            let m = size.half_len();
            let (re, im) = scratch[..2 * m].split_at_mut(m);
            c2r_preprocess_bitrev(spec_re, spec_im, re, im, m);
            cfft_core_scalar(re, im, m, true);
            let mut k = 0;
            while k < m {
                time_out[2 * k] = re[k];
                time_out[2 * k + 1] = im[k];
                k += 1;
            }
        } else {
            dispatch!(
                self.backend.level(),
                simd => rfft_inverse_simd_impl(simd, size, spec_re, spec_im, time_out, scratch)
            );
        }
        Ok(())
    }
}
