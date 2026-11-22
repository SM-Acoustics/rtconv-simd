//! Embedded `#![no_std]`, zero-allocation SIMD FFT backend (`SimdFixedFftBackend`)
//! for `N = 64, 128, 256` using `fearless_simd 1.2.0`.
//!
//! ## Design
//! - Computes `N`-point RFFT and `N`-point normalized C2R IFFT via an inner
//!   `M = N / 2`-point split-complex (SoA) DIT FFT (`M = 32, 64, 128`), halving
//!   the complex transform work compared to a full `N`-point complex IFFT.
//! - Uses caller-provided external scratch (`N` floats = `re[0..M]` + `im[0..M]`),
//!   avoiding large stack allocations (unlike `MicroFftBackend`'s 2 KiB stack buffer).
//! - Precomputed `const` twiddle and bit-reversal tables (`simd_fft_tables.rs`);
//!   zero runtime trig, zero `alloc`, zero `unsafe`.
//! - Stages 1 & 2 (`half = 1, 2`) are fused into a multiplication-free radix-4 pass.
//! - Stages `half = 4, 8, 16, 32, 64` use explicit `fearless_simd 1.2.0` vector
//!   butterflies (`f32x4` for `half = 4` and native `S::f32s` for `half >= LEN`).
//! - Falls back cleanly to scalar butterflies when `SimdBackend::scalar()` is
//!   selected or when the target has no SIMD support (e.g. RISC-V).

use crate::SimdBackend;
use crate::error::ConvError;
use crate::fft::{FftBackend, FftSize};
use crate::simd_fft_tables::{
    BITREV_32, BITREV_64, BITREV_128, R2C_COS_256, R2C_SIN_256, TW_COS_4, TW_COS_8, TW_COS_16,
    TW_COS_32, TW_COS_64, TW_SIN_4, TW_SIN_8, TW_SIN_16, TW_SIN_32, TW_SIN_64,
};
use fearless_simd::{Simd, dispatch, f32x4, prelude::*};
use fearless_simd_macros::simd;

/// Allocation-free, `#![no_std]` SIMD FFT backend for `N = 64, 128, 256`.
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

/// Scalar butterfly stage for `half >= 4`.
#[inline(always)]
fn stage_scalar(
    re: &mut [f32],
    im: &mut [f32],
    m: usize,
    half: usize,
    tw_cos: &[f32],
    tw_sin: &[f32],
    inverse: bool,
) {
    let step = half * 2;
    let mut base = 0;
    if !inverse {
        while base < m {
            let mut j = 0;
            while j < half {
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = re[i0];
                let ui = im[i0];
                let vr = re[i1];
                let vi = im[i1];
                let wr = tw_cos[j];
                let wi = tw_sin[j];
                // t = v * (wr - j*wi)
                let tr = vr * wr + vi * wi;
                let ti = vi * wr - vr * wi;
                re[i0] = ur + tr;
                im[i0] = ui + ti;
                re[i1] = ur - tr;
                im[i1] = ui - ti;
                j += 1;
            }
            base += step;
        }
    } else {
        while base < m {
            let mut j = 0;
            while j < half {
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = re[i0];
                let ui = im[i0];
                let vr = re[i1];
                let vi = im[i1];
                let wr = tw_cos[j];
                let wi = tw_sin[j];
                // t = v * (wr + j*wi)
                let tr = vr * wr - vi * wi;
                let ti = vr * wi + vi * wr;
                re[i0] = ur + tr;
                im[i0] = ui + ti;
                re[i1] = ur - tr;
                im[i1] = ui - ti;
                j += 1;
            }
            base += step;
        }
    }
}

/// 4-wide SIMD butterfly stage (`half == 4`), works cleanly on SSE2, AVX2, NEON, WASM.
#[inline(always)]
fn stage_half4_simd<S: Simd>(simd: S, re: &mut [f32], im: &mut [f32], m: usize, inverse: bool) {
    let wr = f32x4::from_slice(simd, &TW_COS_4);
    let wi = f32x4::from_slice(simd, &TW_SIN_4);
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

/// Native-width SIMD butterfly stage for `half >= 8` (exact multiple of `S::f32s::LEN` on SSE2/AVX2/NEON/WASM).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn stage_native_simd<S: Simd>(
    simd: S,
    re: &mut [f32],
    im: &mut [f32],
    m: usize,
    half: usize,
    tw_cos: &[f32],
    tw_sin: &[f32],
    inverse: bool,
) {
    let lanes = S::f32s::LEN;
    if half < lanes || !half.is_multiple_of(lanes) {
        stage_scalar(re, im, m, half, tw_cos, tw_sin, inverse);
        return;
    }
    let step = half * 2;
    let mut base = 0;
    if !inverse {
        while base < m {
            let mut j = 0;
            while j < half {
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = S::f32s::from_slice(simd, &re[i0..i0 + lanes]);
                let ui = S::f32s::from_slice(simd, &im[i0..i0 + lanes]);
                let vr = S::f32s::from_slice(simd, &re[i1..i1 + lanes]);
                let vi = S::f32s::from_slice(simd, &im[i1..i1 + lanes]);
                let wr = S::f32s::from_slice(simd, &tw_cos[j..j + lanes]);
                let wi = S::f32s::from_slice(simd, &tw_sin[j..j + lanes]);

                let tr = vr.mul_add(wr, vi * wi);
                let ti = vi.mul_add(wr, -(vr * wi));

                (ur + tr).store_slice(&mut re[i0..i0 + lanes]);
                (ui + ti).store_slice(&mut im[i0..i0 + lanes]);
                (ur - tr).store_slice(&mut re[i1..i1 + lanes]);
                (ui - ti).store_slice(&mut im[i1..i1 + lanes]);
                j += lanes;
            }
            base += step;
        }
    } else {
        while base < m {
            let mut j = 0;
            while j < half {
                let i0 = base + j;
                let i1 = i0 + half;
                let ur = S::f32s::from_slice(simd, &re[i0..i0 + lanes]);
                let ui = S::f32s::from_slice(simd, &im[i0..i0 + lanes]);
                let vr = S::f32s::from_slice(simd, &re[i1..i1 + lanes]);
                let vi = S::f32s::from_slice(simd, &im[i1..i1 + lanes]);
                let wr = S::f32s::from_slice(simd, &tw_cos[j..j + lanes]);
                let wi = S::f32s::from_slice(simd, &tw_sin[j..j + lanes]);

                let tr = vr.mul_add(wr, -(vi * wi));
                let ti = vr.mul_add(wi, vi * wr);

                (ur + tr).store_slice(&mut re[i0..i0 + lanes]);
                (ui + ti).store_slice(&mut im[i0..i0 + lanes]);
                (ur - tr).store_slice(&mut re[i1..i1 + lanes]);
                (ui - ti).store_slice(&mut im[i1..i1 + lanes]);
                j += lanes;
            }
            base += step;
        }
    }
}

#[inline(always)]
fn cfft_core_scalar(re: &mut [f32], im: &mut [f32], m: usize, inverse: bool) {
    radix4_stage01(re, im, m, inverse);
    stage_scalar(re, im, m, 4, &TW_COS_4, &TW_SIN_4, inverse);
    stage_scalar(re, im, m, 8, &TW_COS_8, &TW_SIN_8, inverse);
    stage_scalar(re, im, m, 16, &TW_COS_16, &TW_SIN_16, inverse);
    if m >= 64 {
        stage_scalar(re, im, m, 32, &TW_COS_32, &TW_SIN_32, inverse);
    }
    if m >= 128 {
        stage_scalar(re, im, m, 64, &TW_COS_64, &TW_SIN_64, inverse);
    }
}

#[inline(always)]
fn cfft_core_simd<S: Simd>(simd: S, re: &mut [f32], im: &mut [f32], m: usize, inverse: bool) {
    radix4_stage01(re, im, m, inverse);
    stage_half4_simd(simd, re, im, m, inverse);
    stage_native_simd(simd, re, im, m, 8, &TW_COS_8, &TW_SIN_8, inverse);
    stage_native_simd(simd, re, im, m, 16, &TW_COS_16, &TW_SIN_16, inverse);
    if m >= 64 {
        stage_native_simd(simd, re, im, m, 32, &TW_COS_32, &TW_SIN_32, inverse);
    }
    if m >= 128 {
        stage_native_simd(simd, re, im, m, 64, &TW_COS_64, &TW_SIN_64, inverse);
    }
}

#[inline(always)]
fn r2c_untangle(
    z_re: &[f32],
    z_im: &[f32],
    spec_re: &mut [f32],
    spec_im: &mut [f32],
    m: usize,
    r2c_cos: &[f32],
    r2c_sin: &[f32],
) {
    // k = 0 (DC) and k = M (Nyquist)
    spec_re[0] = z_re[0] + z_im[0];
    spec_im[0] = 0.0;
    spec_re[m] = z_re[0] - z_im[0];
    spec_im[m] = 0.0;

    // k = M/2
    let half_m = m / 2;
    spec_re[half_m] = z_re[half_m];
    spec_im[half_m] = -z_im[half_m];

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

        let wc = r2c_cos[k];
        let ws = r2c_sin[k];

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
#[allow(clippy::too_many_arguments)]
fn c2r_preprocess_bitrev(
    spec_re: &[f32],
    spec_im: &[f32],
    re: &mut [f32],
    im: &mut [f32],
    m: usize,
    bitrev: &[u8],
    r2c_cos: &[f32],
    r2c_sin: &[f32],
) {
    // Fold 1/N = 0.5/M directly into the pre-process pass.
    let inv_n = 0.5 / (m as f32);
    let inv_m = 1.0 / (m as f32);

    // k = 0 -> bitrev[0] = 0
    re[0] = inv_n * (spec_re[0] + spec_re[m]);
    im[0] = inv_n * (spec_re[0] - spec_re[m]);

    // k = M/2
    let half_m = m / 2;
    let br_mid = bitrev[half_m] as usize;
    re[br_mid] = inv_m * spec_re[half_m];
    im[br_mid] = -inv_m * spec_im[half_m];

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

        let wc = r2c_cos[k];
        let ws = r2c_sin[k];

        // conj(W_N^k) * D = (wc + j*ws) * (d_re + j*d_im)
        let o_re = wc * d_re - ws * d_im;
        let o_im = wc * d_im + ws * d_re;

        let br_k = bitrev[k] as usize;
        let br_mk = bitrev[mk] as usize;
        re[br_k] = e_re - o_im;
        im[br_k] = e_im + o_re;
        re[br_mk] = e_re + o_im;
        im[br_mk] = -e_im + o_re;
        k += 1;
    }
}

#[inline(always)]
fn tables_for(size: FftSize) -> (usize, &'static [u8], &'static [f32], &'static [f32]) {
    match size {
        FftSize::N64 => (32, &BITREV_32, &TW_COS_32, &TW_SIN_32),
        FftSize::N128 => (64, &BITREV_64, &TW_COS_64, &TW_SIN_64),
        FftSize::N256 => (128, &BITREV_128, &R2C_COS_256, &R2C_SIN_256),
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
    let (m, bitrev, r2c_cos, r2c_sin) = tables_for(size);
    let (re, im) = scratch[..2 * m].split_at_mut(m);

    let mut k = 0;
    while k < m {
        let br = bitrev[k] as usize;
        re[br] = time_in[2 * k];
        im[br] = time_in[2 * k + 1];
        k += 1;
    }

    cfft_core_simd(simd, re, im, m, false);
    r2c_untangle(re, im, spec_re, spec_im, m, r2c_cos, r2c_sin);
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
    let (m, bitrev, r2c_cos, r2c_sin) = tables_for(size);
    let (re, im) = scratch[..2 * m].split_at_mut(m);

    c2r_preprocess_bitrev(spec_re, spec_im, re, im, m, bitrev, r2c_cos, r2c_sin);
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
    fn supports_size(&self, size: FftSize) -> bool {
        matches!(size, FftSize::N64 | FftSize::N128 | FftSize::N256)
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
            let (m, bitrev, r2c_cos, r2c_sin) = tables_for(size);
            let (re, im) = scratch[..2 * m].split_at_mut(m);
            let mut k = 0;
            while k < m {
                let br = bitrev[k] as usize;
                re[br] = time_in[2 * k];
                im[br] = time_in[2 * k + 1];
                k += 1;
            }
            cfft_core_scalar(re, im, m, false);
            r2c_untangle(re, im, spec_re, spec_im, m, r2c_cos, r2c_sin);
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
            let (m, bitrev, r2c_cos, r2c_sin) = tables_for(size);
            let (re, im) = scratch[..2 * m].split_at_mut(m);
            c2r_preprocess_bitrev(spec_re, spec_im, re, im, m, bitrev, r2c_cos, r2c_sin);
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
