//! Allocation-free, `no_std` FFT backend abstraction (`FftBackend`) and
//! primary [`MicroFftBackend`] powered by `microfft 0.6.0`.
//!
//! Spectra are represented in split-complex (SoA) non-redundant format:
//! - `spec_re` and `spec_im` each have length `N / 2 + 1` (`FftSize::spectrum_bins()`).
//! - Bin `0` is DC (`spec_im[0] == 0.0`).
//! - Bin `N / 2` is Nyquist (`spec_im[N / 2] == 0.0`).
//! - Bins `1..N/2` are the positive-frequency complex bins.
//!
//! Because `spec_im[0] == 0.0` and `spec_im[N/2] == 0.0`, every bin `0..=N/2`
//! obeys the standard complex multiply-accumulate rule without special cases,
//! enabling contiguous unit-stride SIMD CMAC over `0..N/2` plus 1 scalar Nyquist bin.

use crate::error::ConvError;
use microfft::Complex32;

/// Supported power-of-two real FFT sizes for the core `no_std` engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftSize {
    /// 64-point real FFT (33 complex bins, block size 32 in 50% overlap-save).
    N64 = 64,
    /// 128-point real FFT (65 complex bins, block size 64 in 50% overlap-save).
    N128 = 128,
    /// 256-point real FFT (129 complex bins, block size 128 in 50% overlap-save).
    N256 = 256,
}

impl FftSize {
    /// Converts a raw FFT length into [`FftSize`].
    #[inline]
    pub const fn from_len(len: usize) -> Result<Self, ConvError> {
        match len {
            64 => Ok(Self::N64),
            128 => Ok(Self::N128),
            256 => Ok(Self::N256),
            _ => Err(ConvError::UnsupportedFftSize { fft_size: len }),
        }
    }

    /// Converts an overlap-save block size (`B = N / 2`) into [`FftSize`].
    #[inline]
    pub const fn from_block_size(block_size: usize) -> Result<Self, ConvError> {
        match block_size {
            32 => Ok(Self::N64),
            64 => Ok(Self::N128),
            128 => Ok(Self::N256),
            _ => Err(ConvError::InvalidBlockSize { block_size }),
        }
    }

    /// Full FFT length `N` in real time-domain samples.
    #[inline]
    pub const fn len(self) -> usize {
        self as usize
    }

    /// Half FFT length `N / 2` (equals the 50% overlap-save block size `B`).
    #[inline]
    pub const fn half_len(self) -> usize {
        self.len() / 2
    }

    /// Number of non-redundant complex frequency bins (`N / 2 + 1`: DC, `1..N/2`, Nyquist).
    #[inline]
    pub const fn spectrum_bins(self) -> usize {
        self.half_len() + 1
    }
}

/// Core `no_std`, allocation-free FFT backend abstraction.
///
/// Designed so embedded targets use [`MicroFftBackend`] with zero `std`/`alloc`,
/// while desktop targets can plug in optional SIMD FFT backends (such as PhastFT
/// behind a feature flag) without altering the core convolver algorithms.
pub trait FftBackend {
    /// Human-readable backend name.
    fn name(&self) -> &'static str;

    /// Whether the FFT/IFFT transform itself uses SIMD instructions.
    ///
    /// Note: Even when this returns `false` (e.g., `microfft`), spectral CMAC
    /// in [`crate::PartitionedConvolver`] still uses `fearless_simd`.
    fn is_simd_accelerated(&self) -> bool;

    /// Returns `true` if `size` is supported by this backend.
    fn supports_size(&self, size: FftSize) -> bool;

    /// Required `f32` scratch buffer length for `size` (may be `0` if stack-only).
    fn scratch_len_f32(&self, size: FftSize) -> usize;

    /// Computes forward real-to-complex FFT (`time_in[0..N]` -> `spec_re[0..N/2+1]`, `spec_im[0..N/2+1]`).
    fn rfft_forward(
        &self,
        size: FftSize,
        time_in: &[f32],
        spec_re: &mut [f32],
        spec_im: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), ConvError>;

    /// Computes normalized inverse complex-to-real FFT (`spec_re[0..N/2+1]`, `spec_im[0..N/2+1]` -> `time_out[0..N]`).
    ///
    /// Output must be normalized by `1 / N` so `rfft_inverse(rfft_forward(x)) ≈ x`.
    fn rfft_inverse(
        &self,
        size: FftSize,
        spec_re: &[f32],
        spec_im: &[f32],
        time_out: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), ConvError>;
}

/// Primary `no_std`, allocation-free FFT backend backed by `microfft 0.6.0`.
///
/// Uses scalar radix-2 in-place transforms (`microfft::real::rfft_*` and
/// `microfft::inverse::ifft_*`). `microfft::inverse::ifft_*` applies `1 / N`
/// normalization internally.
#[derive(Clone, Copy, Debug, Default)]
pub struct MicroFftBackend;

impl MicroFftBackend {
    /// Creates a new `MicroFftBackend` instance.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

#[inline]
fn unpack_microfft_rfft<const HALF: usize>(
    packed: &[Complex32; HALF],
    spec_re: &mut [f32],
    spec_im: &mut [f32],
) {
    // microfft packs DC in packed[0].re and Nyquist in packed[0].im.
    spec_re[0] = packed[0].re;
    spec_im[0] = 0.0;
    spec_re[HALF] = packed[0].im;
    spec_im[HALF] = 0.0;

    let mut k = 1;
    while k < HALF {
        spec_re[k] = packed[k].re;
        spec_im[k] = packed[k].im;
        k += 1;
    }
}

#[inline]
fn pack_hermitian_full<const N: usize>(
    spec_re: &[f32],
    spec_im: &[f32],
    full: &mut [Complex32; N],
) {
    let half = N / 2;
    full[0] = Complex32::new(spec_re[0], 0.0);
    full[half] = Complex32::new(spec_re[half], 0.0);

    let mut k = 1;
    while k < half {
        let re = spec_re[k];
        let im = spec_im[k];
        full[k] = Complex32::new(re, im);
        full[N - k] = Complex32::new(re, -im);
        k += 1;
    }
}

impl FftBackend for MicroFftBackend {
    #[inline]
    fn name(&self) -> &'static str {
        "microfft-0.6.0 (radix-2 scalar)"
    }

    #[inline]
    fn is_simd_accelerated(&self) -> bool {
        false
    }

    #[inline]
    fn supports_size(&self, size: FftSize) -> bool {
        matches!(size, FftSize::N64 | FftSize::N128 | FftSize::N256)
    }

    #[inline]
    fn scratch_len_f32(&self, _size: FftSize) -> usize {
        // Uses small bounded fixed-size stack buffers (<= 2 KiB for N=256).
        0
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

        match size {
            FftSize::N64 => {
                let mut buf = [0.0f32; 64];
                buf.copy_from_slice(time_in);
                let packed = microfft::real::rfft_64(&mut buf);
                unpack_microfft_rfft::<32>(packed, spec_re, spec_im);
            }
            FftSize::N128 => {
                let mut buf = [0.0f32; 128];
                buf.copy_from_slice(time_in);
                let packed = microfft::real::rfft_128(&mut buf);
                unpack_microfft_rfft::<64>(packed, spec_re, spec_im);
            }
            FftSize::N256 => {
                let mut buf = [0.0f32; 256];
                buf.copy_from_slice(time_in);
                let packed = microfft::real::rfft_256(&mut buf);
                unpack_microfft_rfft::<128>(packed, spec_re, spec_im);
            }
        }
        Ok(())
    }

    fn rfft_inverse(
        &self,
        size: FftSize,
        spec_re: &[f32],
        spec_im: &[f32],
        time_out: &mut [f32],
        _scratch: &mut [f32],
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

        match size {
            FftSize::N64 => {
                let mut buf = [Complex32::new(0.0, 0.0); 64];
                pack_hermitian_full::<64>(spec_re, spec_im, &mut buf);
                let out = microfft::inverse::ifft_64(&mut buf);
                let mut i = 0;
                while i < 64 {
                    time_out[i] = out[i].re;
                    i += 1;
                }
            }
            FftSize::N128 => {
                let mut buf = [Complex32::new(0.0, 0.0); 128];
                pack_hermitian_full::<128>(spec_re, spec_im, &mut buf);
                let out = microfft::inverse::ifft_128(&mut buf);
                let mut i = 0;
                while i < 128 {
                    time_out[i] = out[i].re;
                    i += 1;
                }
            }
            FftSize::N256 => {
                let mut buf = [Complex32::new(0.0, 0.0); 256];
                pack_hermitian_full::<256>(spec_re, spec_im, &mut buf);
                let out = microfft::inverse::ifft_256(&mut buf);
                let mut i = 0;
                while i < 256 {
                    time_out[i] = out[i].re;
                    i += 1;
                }
            }
        }
        Ok(())
    }
}
