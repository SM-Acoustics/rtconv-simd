//! Uniform Partitioned Frequency-Domain Delay Line (FDL) convolution engine
//! using 50% Overlap-Save (OLS).
//!
//! ## Choice of Overlap-Save (OLS) over Overlap-Add (OLA)
//! - Each IR partition of length `B` (`block_size`) is zero-padded to `N = 2B`
//!   and pre-transformed once during `prepare()`.
//! - Each input block of length `B` is concatenated with the previous block
//!   `[prev_block, cur_block]` to form an `N`-sample frame, transformed once,
//!   and pushed into the FDL ring buffer.
//! - After SIMD complex multiply-accumulate across all `P` partitions and a
//!   single normalized `N`-point IFFT, the first `B` samples (circular aliasing)
//!   are discarded and the second `B` samples `time_buf[B..2B]` are the exact
//!   linear convolution output.
//! - Compared to OLA, OLS eliminates the output overlap-add state buffer and
//!   post-IFFT summation pass, reducing memory traffic and simplifying RT bounds.

use crate::error::ConvError;
use crate::fft::{FftBackend, FftSize, MicroFftBackend};
use crate::simd_kernels::{cmac_fdl_scalar, cmac_fdl_simd};
use crate::{Capabilities, EngineKind, SimdBackend};

/// Configuration for [`PartitionedConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartitionedConfig {
    /// Processing block size `B` in samples (`32`, `64`, or `128`).
    /// Corresponding FFT size is `N = 2 * B` (`64`, `128`, or `256`).
    pub block_size: usize,
    /// Maximum supported impulse response length in samples.
    pub max_ir_len: usize,
}

/// Checked workspace layout and memory requirements for [`PartitionedConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartitionedMemoryRequirements {
    /// Processing block size `B`.
    pub block_size: usize,
    /// FFT size `N = 2 * B`.
    pub fft_size: FftSize,
    /// Non-redundant complex bins `N / 2 + 1`.
    pub spectrum_bins: usize,
    /// Maximum number of uniform partitions `ceil(max_ir_len / B)`.
    pub max_partitions: usize,
    /// `f32` elements for precomputed IR partition spectra (real + imag).
    pub ir_spectra_f32: usize,
    /// `f32` elements for FDL input spectra ring buffer (real + imag).
    pub fdl_spectra_f32: usize,
    /// `f32` elements for previous input block, time-domain FFT frame, accumulator, and FFT scratch.
    pub scratch_and_state_f32: usize,
    /// Total `f32` elements required in caller workspace.
    pub total_f32: usize,
    /// Total workspace size in bytes.
    pub total_bytes: usize,
}

/// Zero-allocation, `no_std` uniform partitioned FDL convolver.
pub struct PartitionedConvolver<'a, F: FftBackend = MicroFftBackend> {
    fft: F,
    backend: SimdBackend,
    block_size: usize,
    fft_size: FftSize,
    bins: usize,
    max_ir_len: usize,
    active_ir_len: usize,
    num_partitions: usize,
    fdl_head: usize,

    // Partitioned IR spectra: each has `max_partitions * bins` elements.
    ir_re: &'a mut [f32],
    ir_im: &'a mut [f32],
    // FDL ring buffer of input spectra: each has `max_partitions * bins` elements.
    fdl_re: &'a mut [f32],
    fdl_im: &'a mut [f32],
    // Previous block of `B` real samples for 50% overlap-save.
    prev_input: &'a mut [f32],
    // Reusable `N = 2B` time-domain scratch buffer.
    time_buf: &'a mut [f32],
    // Reusable spectral accumulator (`bins` each).
    acc_re: &'a mut [f32],
    acc_im: &'a mut [f32],
    // Optional FFT backend scratch.
    fft_scratch: &'a mut [f32],
}

impl<'a, F: FftBackend> PartitionedConvolver<'a, F> {
    /// Computes checked workspace requirements for `config` and `fft`.
    pub fn memory_requirements(
        config: PartitionedConfig,
        fft: &F,
    ) -> Result<PartitionedMemoryRequirements, ConvError> {
        if config.max_ir_len == 0 {
            return Err(ConvError::EmptyIr);
        }
        let fft_size = FftSize::from_block_size(config.block_size)?;
        if !fft.supports_size(fft_size) {
            return Err(ConvError::UnsupportedFftSize {
                fft_size: fft_size.len(),
            });
        }

        let b = config.block_size;
        let n = fft_size.len();
        let bins = fft_size.spectrum_bins();
        let max_partitions = config
            .max_ir_len
            .checked_add(b - 1)
            .ok_or(ConvError::ArithmeticOverflow)?
            / b;

        let part_plane = max_partitions
            .checked_mul(bins)
            .ok_or(ConvError::ArithmeticOverflow)?;
        let ir_spectra_f32 = part_plane
            .checked_mul(2)
            .ok_or(ConvError::ArithmeticOverflow)?;
        let fdl_spectra_f32 = ir_spectra_f32;

        let fft_scratch_len = fft.scratch_len_f32(fft_size);
        let acc_total = bins.checked_mul(2).ok_or(ConvError::ArithmeticOverflow)?;
        let scratch_and_state_f32 = b
            .checked_add(n)
            .and_then(|v| v.checked_add(acc_total))
            .and_then(|v| v.checked_add(fft_scratch_len))
            .ok_or(ConvError::ArithmeticOverflow)?;

        let total_f32 = ir_spectra_f32
            .checked_add(fdl_spectra_f32)
            .and_then(|v| v.checked_add(scratch_and_state_f32))
            .ok_or(ConvError::ArithmeticOverflow)?;
        let total_bytes = total_f32
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(ConvError::ArithmeticOverflow)?;

        Ok(PartitionedMemoryRequirements {
            block_size: b,
            fft_size,
            spectrum_bins: bins,
            max_partitions,
            ir_spectra_f32,
            fdl_spectra_f32,
            scratch_and_state_f32,
            total_f32,
            total_bytes,
        })
    }

    /// Validates configuration and impulse response, precomputes all partition
    /// spectra inside caller-owned `workspace`, and returns an initialized engine.
    pub fn prepare(
        ir: &[f32],
        config: PartitionedConfig,
        workspace: &'a mut [f32],
        backend: SimdBackend,
        fft: F,
    ) -> Result<Self, ConvError> {
        let req = Self::memory_requirements(config, &fft)?;
        if ir.is_empty() {
            return Err(ConvError::EmptyIr);
        }
        if ir.len() > config.max_ir_len {
            return Err(ConvError::IrTooLong {
                ir_len: ir.len(),
                max_ir_len: config.max_ir_len,
            });
        }
        for (idx, &coeff) in ir.iter().enumerate() {
            if !coeff.is_finite() {
                return Err(ConvError::NonFiniteCoefficient { index: idx });
            }
        }
        if workspace.len() < req.total_f32 {
            return Err(ConvError::WorkspaceTooSmall {
                required: req.total_f32,
                provided: workspace.len(),
            });
        }

        let b = req.block_size;
        let n = req.fft_size.len();
        let bins = req.spectrum_bins;
        let plane = req.max_partitions * bins;
        let fft_scratch_len = fft.scratch_len_f32(req.fft_size);

        let (ir_re, rest) = workspace.split_at_mut(plane);
        let (ir_im, rest) = rest.split_at_mut(plane);
        let (fdl_re, rest) = rest.split_at_mut(plane);
        let (fdl_im, rest) = rest.split_at_mut(plane);
        let (prev_input, rest) = rest.split_at_mut(b);
        let (time_buf, rest) = rest.split_at_mut(n);
        let (acc_re, rest) = rest.split_at_mut(bins);
        let (acc_im, rest) = rest.split_at_mut(bins);
        let (fft_scratch, _) = rest.split_at_mut(fft_scratch_len);

        ir_re.fill(0.0);
        ir_im.fill(0.0);
        fdl_re.fill(0.0);
        fdl_im.fill(0.0);
        prev_input.fill(0.0);
        time_buf.fill(0.0);
        acc_re.fill(0.0);
        acc_im.fill(0.0);
        fft_scratch.fill(0.0);

        let num_partitions = ir.len().div_ceil(b);

        // Precompute zero-padded partition spectra: [h_p[0..chunk], 0..0].
        let mut p = 0;
        while p < num_partitions {
            let start = p * b;
            let end = (start + b).min(ir.len());
            let chunk_len = end - start;

            time_buf.fill(0.0);
            time_buf[..chunk_len].copy_from_slice(&ir[start..end]);

            let off = p * bins;
            fft.rfft_forward(
                req.fft_size,
                time_buf,
                &mut ir_re[off..off + bins],
                &mut ir_im[off..off + bins],
                fft_scratch,
            )?;
            p += 1;
        }
        time_buf.fill(0.0);

        Ok(Self {
            fft,
            backend,
            block_size: b,
            fft_size: req.fft_size,
            bins,
            max_ir_len: config.max_ir_len,
            active_ir_len: ir.len(),
            num_partitions,
            fdl_head: 0,
            ir_re,
            ir_im,
            fdl_re,
            fdl_im,
            prev_input,
            time_buf,
            acc_re,
            acc_im,
            fft_scratch,
        })
    }

    /// Processes exactly `block_size` samples via 50% overlap-save FDL.
    ///
    /// Returns [`ConvError::MismatchedBufferLength`] if `input.len() != block_size`
    /// or `output.len() != block_size`. Never allocates or panics.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), ConvError> {
        let b = self.block_size;
        if input.len() != b {
            return Err(ConvError::MismatchedBufferLength {
                expected: b,
                actual: input.len(),
            });
        }
        if output.len() != b {
            return Err(ConvError::MismatchedBufferLength {
                expected: b,
                actual: output.len(),
            });
        }

        // 1. Build 2B overlap-save input frame: [prev_input, input].
        self.time_buf[..b].copy_from_slice(self.prev_input);
        self.time_buf[b..2 * b].copy_from_slice(input);
        self.prev_input.copy_from_slice(input);

        // 2. Advance FDL ring buffer head and compute forward RFFT into head slot.
        self.fdl_head = if self.fdl_head + 1 == self.num_partitions {
            0
        } else {
            self.fdl_head + 1
        };
        let head_off = self.fdl_head * self.bins;
        self.fft.rfft_forward(
            self.fft_size,
            self.time_buf,
            &mut self.fdl_re[head_off..head_off + self.bins],
            &mut self.fdl_im[head_off..head_off + self.bins],
            self.fft_scratch,
        )?;

        // 3. Zero spectral accumulator and accumulate all P partitions (1 dispatch!).
        self.acc_re.fill(0.0);
        self.acc_im.fill(0.0);
        if self.backend.is_scalar() {
            cmac_fdl_scalar(
                self.acc_re,
                self.acc_im,
                self.fdl_re,
                self.fdl_im,
                self.ir_re,
                self.ir_im,
                self.num_partitions,
                self.bins,
                self.fdl_head,
            );
        } else {
            cmac_fdl_simd(
                self.backend.level(),
                self.acc_re,
                self.acc_im,
                self.fdl_re,
                self.fdl_im,
                self.ir_re,
                self.ir_im,
                self.num_partitions,
                self.bins,
                self.fdl_head,
            );
        }

        // 4. Inverse FFT (normalized by 1/N) and extract valid second half [B..2B].
        self.fft.rfft_inverse(
            self.fft_size,
            self.acc_re,
            self.acc_im,
            self.time_buf,
            self.fft_scratch,
        )?;
        output.copy_from_slice(&self.time_buf[b..2 * b]);

        Ok(())
    }

    /// Resets the FDL ring buffer and overlap state without deallocating memory.
    #[inline]
    pub fn reset(&mut self) {
        self.fdl_re.fill(0.0);
        self.fdl_im.fill(0.0);
        self.prev_input.fill(0.0);
        self.time_buf.fill(0.0);
        self.acc_re.fill(0.0);
        self.acc_im.fill(0.0);
        self.fdl_head = 0;
    }

    /// Internal algorithmic latency in samples for block-aligned `process_block` calls.
    ///
    /// Note: Overlap-save produces `y[n..n+B]` immediately from `[x_prev, x_cur]`
    /// with `0` extra internal buffering samples beyond the host's `B`-sample block
    /// framing. If the host audio callback operates at a smaller block size than `B`,
    /// accumulating `B` samples introduces `B` samples of framing latency (which is
    /// why [`crate::HybridConvolver`] pairs a direct FIR head with a larger FFT tail).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        0
    }

    /// Processing block size `B` in samples.
    #[inline]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Active number of uniform partitions.
    #[inline]
    pub const fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Engine capabilities and runtime metadata.
    #[inline]
    pub fn capabilities(&self) -> Capabilities {
        Capabilities {
            engine: EngineKind::Partitioned,
            simd_backend: self.backend.name(),
            fft_backend: self.fft.name(),
            fft_is_simd: self.fft.is_simd_accelerated(),
            block_size: Some(self.block_size),
            fft_size: Some(self.fft_size.len()),
            max_ir_len: self.max_ir_len,
            active_ir_len: self.active_ir_len,
            latency_samples: 0,
        }
    }
}
