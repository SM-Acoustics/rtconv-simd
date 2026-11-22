//! Optional desktop SIMD FFT backend powered by `phastft 0.4.1` (`default-features = false`).
//!
//! ## Real-Time & Dependency Notes
//! - Gated behind `#[cfg(feature = "phastft")]`; disabled in `--no-default-features`
//!   so the core embedded crate remains strictly `#![no_std]` with zero `alloc`.
//! - `phastft 0.4.1` is configured with `default-features = false` (no `rayon`
//!   thread pool). `Options` explicitly sets `multithreaded_bit_reversal = false`
//!   and `smallest_parallel_chunk_size = usize::MAX`.
//! - Planners (`PlannerR2c32`) for `N = 64, 128, 256` are created once in
//!   [`PhastFftBackend::new()`] BEFORE audio processing.
//! - Uses `r2c_fft_f32_with_planner_and_opts` and `c2r_fft_f32_with_planner_and_opts`
//!   with caller-supplied reusable scratch (`N` floats = `N/2` re + `N/2` im),
//!   guaranteeing zero heap allocations in the audio callback.
//! - Dependency note: `phastft 0.4.1` depends internally on `fearless_simd 0.5.0`,
//!   while `rtconv_simd` uses `fearless_simd 1.2.0`. Cargo compiles both semver-major
//!   versions side-by-side cleanly when `--features phastft` is enabled.

use crate::error::ConvError;
use crate::fft::{FftBackend, FftSize};
use phastft::options::Options;
use phastft::planner::PlannerR2c32;
use phastft::{c2r_fft_f32_with_planner_and_opts, r2c_fft_f32_with_planner_and_opts};

/// Desktop SIMD FFT backend backed by `phastft 0.4.1`.
#[derive(Clone, Debug)]
pub struct PhastFftBackend {
    plan_64: PlannerR2c32,
    plan_128: PlannerR2c32,
    plan_256: PlannerR2c32,
    opts: Options,
}

impl PhastFftBackend {
    /// Precomputes `PlannerR2c32` instances for `N = 64, 128, 256`.
    ///
    /// Call once during initialization (outside the real-time audio callback).
    pub fn new() -> Self {
        let mut opts = Options::default();
        opts.multithreaded_bit_reversal = false;
        opts.smallest_parallel_chunk_size = usize::MAX;
        Self {
            plan_64: PlannerR2c32::new(64),
            plan_128: PlannerR2c32::new(128),
            plan_256: PlannerR2c32::new(256),
            opts,
        }
    }

    #[inline]
    fn planner(&self, size: FftSize) -> &PlannerR2c32 {
        match size {
            FftSize::N64 => &self.plan_64,
            FftSize::N128 => &self.plan_128,
            FftSize::N256 => &self.plan_256,
        }
    }
}

impl Default for PhastFftBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FftBackend for PhastFftBackend {
    #[inline]
    fn name(&self) -> &'static str {
        "phastft-0.4.1 (desktop SIMD R2C/C2R)"
    }

    #[inline]
    fn is_simd_accelerated(&self) -> bool {
        true
    }

    #[inline]
    fn supports_size(&self, size: FftSize) -> bool {
        matches!(size, FftSize::N64 | FftSize::N128 | FftSize::N256)
    }

    #[inline]
    fn scratch_len_f32(&self, size: FftSize) -> usize {
        // C2R requires N/2 scratch_re + N/2 scratch_im = N floats.
        size.len()
    }

    fn rfft_forward(
        &self,
        size: FftSize,
        time_in: &[f32],
        spec_re: &mut [f32],
        spec_im: &mut [f32],
        _scratch: &mut [f32],
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

        r2c_fft_f32_with_planner_and_opts(
            time_in,
            &mut spec_re[..bins],
            &mut spec_im[..bins],
            self.planner(size),
            &self.opts,
        );
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
        let half = size.half_len();
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

        let (scratch_re, scratch_im) = scratch[..n].split_at_mut(half);
        c2r_fft_f32_with_planner_and_opts(
            &spec_re[..bins],
            &spec_im[..bins],
            time_out,
            self.planner(size),
            &self.opts,
            scratch_re,
            scratch_im,
        );
        // PhastFT's `c2r_fft_f32_with_planner_and_opts` already normalizes by 1/N internally.
        Ok(())
    }
}
