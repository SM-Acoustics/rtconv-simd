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

/// Supported power-of-two real FFT sizes for the core `no_std` engine (`N = 64 ..= 16384`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FftSize {
    /// 64-point real FFT (33 complex bins, block size 32 in 50% overlap-save).
    N64 = 64,
    /// 128-point real FFT (65 complex bins, block size 64 in 50% overlap-save).
    N128 = 128,
    /// 256-point real FFT (129 complex bins, block size 128 in 50% overlap-save).
    N256 = 256,
    /// 512-point real FFT (257 complex bins, block size 256 in 50% overlap-save).
    N512 = 512,
    /// 1024-point real FFT (513 complex bins, block size 512 in 50% overlap-save).
    N1024 = 1024,
    /// 2048-point real FFT (1025 complex bins, block size 1024 in 50% overlap-save).
    N2048 = 2048,
    /// 4096-point real FFT (2049 complex bins, block size 2048 in 50% overlap-save).
    N4096 = 4096,
    /// 8192-point real FFT (4097 complex bins, block size 4096 in 50% overlap-save).
    N8192 = 8192,
    /// 16384-point real FFT (8193 complex bins, block size 8192 in 50% overlap-save).
    N16384 = 16384,
}

impl FftSize {
    /// All supported FFT sizes in ascending order (`N64` through `N16384`).
    pub const ALL: [Self; 9] = [
        Self::N64,
        Self::N128,
        Self::N256,
        Self::N512,
        Self::N1024,
        Self::N2048,
        Self::N4096,
        Self::N8192,
        Self::N16384,
    ];

    /// Converts a raw FFT length into [`FftSize`].
    #[inline]
    pub const fn from_len(len: usize) -> Result<Self, ConvError> {
        match len {
            64 => Ok(Self::N64),
            128 => Ok(Self::N128),
            256 => Ok(Self::N256),
            512 => Ok(Self::N512),
            1024 => Ok(Self::N1024),
            2048 => Ok(Self::N2048),
            4096 => Ok(Self::N4096),
            8192 => Ok(Self::N8192),
            16384 => Ok(Self::N16384),
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
            256 => Ok(Self::N512),
            512 => Ok(Self::N1024),
            1024 => Ok(Self::N2048),
            2048 => Ok(Self::N4096),
            4096 => Ok(Self::N8192),
            8192 => Ok(Self::N16384),
            _ => Err(ConvError::InvalidBlockSize { block_size }),
        }
    }

    /// Full FFT length `N` in real time-domain samples.
    #[inline]
    pub const fn len(self) -> usize {
        self as usize
    }

    /// Base-2 logarithm of `N` (`6..=14`).
    #[inline]
    pub const fn log2(self) -> u32 {
        (self as usize).trailing_zeros()
    }

    /// Always `false` since all supported FFT sizes are strictly positive.
    #[inline]
    pub const fn is_empty(self) -> bool {
        false
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
/// Designed so embedded targets use [`MicroFftBackend`] or [`crate::SimdFixedFftBackend`]
/// with zero `std`/`alloc`, while desktop targets can plug in optional SIMD FFT backends
/// (such as PhastFT behind a feature flag) without altering the core convolver algorithms.
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

    /// Required `f32` scratch buffer length for `size`.
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

    /// Returns `true` if this backend supports zero-extra-RAM, in-place time-distributed
    /// transform subdivision via [`Self::rfft_forward_step`] and [`Self::rfft_inverse_step`].
    #[inline]
    fn supports_staged_fft(&self) -> bool {
        false
    }

    /// Number of bounded-cost forward RFFT steps for `size` (`1` for monolithic backends).
    #[inline]
    fn staged_forward_steps(&self, _size: FftSize) -> usize {
        1
    }

    /// Number of bounded-cost inverse C2R IFFT steps for `size` (`1` for monolithic backends).
    #[inline]
    fn staged_inverse_steps(&self, _size: FftSize) -> usize {
        1
    }

    /// Executes one bounded-cost forward RFFT step (`step` in `0..self.staged_forward_steps(size)`).
    ///
    /// - `time_in` (`len == N`) is read on `step == 0` (unused on subsequent steps for staged backends).
    /// - `spec_re` and `spec_im` (`len >= N / 2 + 1`) carry the in-place transform state across
    ///   steps and hold the completed split-complex spectrum after the final step.
    /// - Backends with `supports_staged_fft() == false` fall back to [`Self::rfft_forward`] on `step == 0`.
    fn rfft_forward_step(
        &self,
        size: FftSize,
        step: usize,
        time_in: &[f32],
        spec_re: &mut [f32],
        spec_im: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), ConvError> {
        if step == 0 {
            self.rfft_forward(size, time_in, spec_re, spec_im, scratch)
        } else {
            Err(ConvError::InvalidStateTransition)
        }
    }

    /// Executes one bounded-cost normalized inverse C2R IFFT step
    /// (`step` in `0..self.staged_inverse_steps(size)`).
    ///
    /// - `spec_re` and `spec_im` (`len >= N / 2 + 1`) hold the input spectrum on `step == 0`
    ///   and are transformed in-place across steps.
    /// - `time_out` (`len == N`) is written on the final step
    ///   (`step + 1 == self.staged_inverse_steps(size)`).
    /// - Backends with `supports_staged_fft() == false` fall back to [`Self::rfft_inverse`] on `step == 0`.
    fn rfft_inverse_step(
        &self,
        size: FftSize,
        step: usize,
        spec_re: &mut [f32],
        spec_im: &mut [f32],
        time_out: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), ConvError> {
        if step == 0 {
            self.rfft_inverse(size, spec_re, spec_im, time_out, scratch)
        } else {
            Err(ConvError::InvalidStateTransition)
        }
    }
}

/// Primary `no_std`, allocation-free scalar FFT backend backed by `microfft 0.6.0`.
///
/// Supports all sizes `N64` through `N16384` using `microfft`'s in-place real-input
/// transforms (`microfft::real::rfft_*`, requiring feature `size-16384` -> `16,380 B` sine ROM
/// so `rfft_16384`'s real-recombination stride stays in-bounds) operating exclusively inside the
/// caller-supplied `scratch` slice (`N` `f32` elements) with zero large stack buffers.
///
/// For `rfft_inverse`, a Hermitian-symmetric spectrum `X[k] = R[k] + j*I[k]` is mapped to the
/// real sequence `u[k] = R[k] + I[k]`, `u[N - k] = R[k] - I[k]`. Because `R` is even and `I`
/// is odd, the forward real DFT `U = RFFT(u)` satisfies:
/// `Re(U[n]) = sum R[k] cos(2*pi*k*n/N)` and `Im(U[n]) = -sum I[k] sin(2*pi*k*n/N)`, so
/// `x[n] = (Re(U[n]) + Im(U[n])) / N` and `x[N - n] = (Re(U[n]) - Im(U[n])) / N`.
/// This computes the inverse transform via an `N/2`-point inner CFFT (halving work compared
/// to `microfft::inverse::ifft_*`) while keeping peak stack usage minimal (`O(log N)` frames).
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
fn pack_hermitian_real(spec_re: &[f32], spec_im: &[f32], buf: &mut [f32], n: usize) {
    let half = n / 2;
    buf[0] = spec_re[0];
    buf[half] = spec_re[half];
    let mut k = 1;
    while k < half {
        let re = spec_re[k];
        let im = spec_im[k];
        buf[k] = re + im;
        buf[n - k] = re - im;
        k += 1;
    }
}

#[inline]
fn unpack_inverse_from_rfft<const HALF: usize>(packed: &[Complex32; HALF], time_out: &mut [f32]) {
    let n = HALF * 2;
    let inv_n = 1.0 / (n as f32);
    time_out[0] = packed[0].re * inv_n;
    time_out[HALF] = packed[0].im * inv_n;
    let mut k = 1;
    while k < HALF {
        let re = packed[k].re;
        let im = packed[k].im;
        time_out[k] = (re + im) * inv_n;
        time_out[n - k] = (re - im) * inv_n;
        k += 1;
    }
}

macro_rules! dispatch_microfft_forward {
    ($size:expr, $scratch:expr, $spec_re:expr, $spec_im:expr, $( $var:ident => ($n:expr, $half:expr, $func:ident) ),+ $(,)?) => {
        match $size {
            $(
                FftSize::$var => {
                    let Ok(buf): Result<&mut [f32; $n], _> = (&mut $scratch[..$n]).try_into() else {
                        return Err(ConvError::WorkspaceTooSmall {
                            required: $n,
                            provided: $scratch.len(),
                        });
                    };
                    let packed = microfft::real::$func(buf);
                    unpack_microfft_rfft::<$half>(packed, $spec_re, $spec_im);
                }
            )+
        }
    };
}

macro_rules! dispatch_microfft_inverse {
    ($size:expr, $scratch:expr, $time_out:expr, $( $var:ident => ($n:expr, $half:expr, $func:ident) ),+ $(,)?) => {
        match $size {
            $(
                FftSize::$var => {
                    let Ok(buf): Result<&mut [f32; $n], _> = (&mut $scratch[..$n]).try_into() else {
                        return Err(ConvError::WorkspaceTooSmall {
                            required: $n,
                            provided: $scratch.len(),
                        });
                    };
                    let packed = microfft::real::$func(buf);
                    unpack_inverse_from_rfft::<$half>(packed, $time_out);
                }
            )+
        }
    };
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
    fn supports_size(&self, _size: FftSize) -> bool {
        true
    }

    #[inline]
    fn scratch_len_f32(&self, size: FftSize) -> usize {
        // Operates strictly inside caller scratch (`N` floats), avoiding `[f32; N]` and
        // `[Complex32; N]` stack buffers on embedded targets.
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

        scratch[..n].copy_from_slice(time_in);
        dispatch_microfft_forward!(
            size,
            scratch,
            spec_re,
            spec_im,
            N64 => (64, 32, rfft_64),
            N128 => (128, 64, rfft_128),
            N256 => (256, 128, rfft_256),
            N512 => (512, 256, rfft_512),
            N1024 => (1024, 512, rfft_1024),
            N2048 => (2048, 1024, rfft_2048),
            N4096 => (4096, 2048, rfft_4096),
            N8192 => (8192, 4096, rfft_8192),
            N16384 => (16384, 8192, rfft_16384),
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

        pack_hermitian_real(spec_re, spec_im, &mut scratch[..n], n);
        dispatch_microfft_inverse!(
            size,
            scratch,
            time_out,
            N64 => (64, 32, rfft_64),
            N128 => (128, 64, rfft_128),
            N256 => (256, 128, rfft_256),
            N512 => (512, 256, rfft_512),
            N1024 => (1024, 512, rfft_1024),
            N2048 => (2048, 1024, rfft_2048),
            N4096 => (4096, 2048, rfft_4096),
            N8192 => (8192, 4096, rfft_8192),
            N16384 => (16384, 8192, rfft_16384),
        );
        Ok(())
    }
}
