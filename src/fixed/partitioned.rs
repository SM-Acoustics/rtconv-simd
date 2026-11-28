//! Pure Rust `#![no_std]` Q1.15 Block-Floating-Point uniform partitioned FDL convolver
//! (`FixedQ15PartitionedConvolver`).
//!
//! Zero heap allocation, `#![forbid(unsafe_code)]`, FPU-free integer arithmetic with exact
//! per-partition BFP exponent alignment and saturation telemetry.
//!
//! Transforms use the 32-bit-mantissa BFP real FFT core (`crate::fixed::fft_q31`, `N/2`
//! complex FFT with real-input packing); spectra are stored as Q1.15 BFP and the accumulated
//! spectrum is inverse-transformed from `i32` mantissas, so the only 16-bit roundings are the
//! partition spectra and the output sample (see `crate::fixed::spectral`).

use crate::fixed::block_float::{Q31Taps, TapSource};
use crate::fixed::fft_q15::fixed_fft_supports_size;
use crate::fixed::fft_q31::{
    Q31FftState, acc_i64_to_q31_spectrum_in_place, q31_fft_steps, q31_irfft_sample, q31_irfft_step,
    q31_rfft_step, q31_spectrum_to_q15, scale_rne_sat,
};
use crate::fixed::spectral::{
    cmac_guard_bits, cmac_partition, exp_to_word, irfft_buf_i16, prepare_partition, rfft_buf_i16,
};
const _: () = assert!(irfft_buf_i16(2048) <= 8 * (2048 / 2 + 1));
use crate::{ConvError, FftSize};

/// Configuration for [`FixedQ15PartitionedConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ15PartitionedConfig {
    /// Processing block size `B` (`32`, `64`, `128`, `256`, `512`, or `1024`, corresponding to
    /// FFT sizes `N64` through `N2048`). Larger block sizes are rejected with
    /// [`ConvError::UnsupportedFftSize`].
    pub block_size: usize,
    /// Maximum supported impulse response length in taps.
    pub max_ir_len: usize,
}

/// Caller-supplied `&mut [i16]` workspace requirements for [`FixedQ15PartitionedConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ15PartitionedMemoryRequirements {
    /// Number of uniform partitions `ceil(max_ir_len / block_size)`.
    pub num_partitions: usize,
    /// FFT length `N = 2 * block_size`.
    pub fft_size: usize,
    /// Total required `&mut [i16]` elements.
    pub total_i16: usize,
    /// Total required workspace size in bytes (`total_i16 * 2`).
    pub total_bytes: usize,
}

/// Runtime metadata for [`FixedQ15PartitionedConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ15PartitionedCapabilities {
    /// Processing block size `B`.
    pub block_size: usize,
    /// FFT length `N = 2 * B`.
    pub fft_size: usize,
    /// Maximum configured impulse response length.
    pub max_ir_len: usize,
    /// Active impulse response length.
    pub active_ir_len: usize,
    /// Active partition count.
    pub active_partitions: usize,
    /// Algorithmic latency in samples (equals `0` per block-aligned overlap-save output).
    pub latency_samples: usize,
}

/// Uniform partitioned overlap-save FDL convolver in Q1.15 Block Floating Point (`#![no_std]`).
pub struct FixedQ15PartitionedConvolver<'a> {
    workspace: &'a mut [i16],
    fft_size: FftSize,
    block_size: usize,
    bins: usize,
    num_partitions: usize,
    active_partitions: usize,
    max_ir_len: usize,
    active_ir_len: usize,
    fdl_head: usize,
    saturation_count: u64,
    // Flat slice offsets into `workspace`:
    overlap_in_off: usize,
    ir_re_off: usize,
    ir_im_off: usize,
    fdl_re_off: usize,
    fdl_im_off: usize,
    ir_exp_off: usize,
    fdl_exp_off: usize,
    acc_off: usize,
    fft_scratch_off: usize,
}

impl<'a> FixedQ15PartitionedConvolver<'a> {
    /// Computes the exact `&mut [i16]` workspace requirements for `config`.
    pub fn memory_requirements(
        config: FixedQ15PartitionedConfig,
    ) -> Result<FixedQ15PartitionedMemoryRequirements, ConvError> {
        if config.max_ir_len == 0 {
            return Err(ConvError::EmptyIr);
        }
        let fft = FftSize::from_block_size(config.block_size)?;
        if !fixed_fft_supports_size(fft) {
            return Err(ConvError::UnsupportedFftSize {
                fft_size: fft.len(),
            });
        }
        let n = fft.len();
        let bins = fft.spectrum_bins();
        let num_partitions = config.max_ir_len.div_ceil(config.block_size);
        let part_bins = num_partitions
            .checked_mul(bins)
            .ok_or(ConvError::ArithmeticOverflow)?;
        let fft_sc = rfft_buf_i16(n);
        debug_assert!(irfft_buf_i16(n) <= 8 * bins);

        // Layout:
        // overlap_in(N) + ir_re(P*bins) + ir_im(P*bins) + fdl_re(P*bins) + fdl_im(P*bins)
        // + ir_exp(P) + fdl_exp(P) + acc(8*bins: i64 re/im accumulators, then the in-place
        // 4N+4-word inverse transform) + forward transform buffer (2N+4)
        let total_i16 = n
            .checked_add(
                part_bins
                    .checked_mul(4)
                    .ok_or(ConvError::ArithmeticOverflow)?,
            )
            .and_then(|v| v.checked_add(num_partitions.checked_mul(2)?))
            .and_then(|v| v.checked_add(bins.checked_mul(8)?))
            .and_then(|v| v.checked_add(fft_sc))
            .ok_or(ConvError::ArithmeticOverflow)?;
        let total_bytes = total_i16
            .checked_mul(core::mem::size_of::<i16>())
            .ok_or(ConvError::ArithmeticOverflow)?;

        Ok(FixedQ15PartitionedMemoryRequirements {
            num_partitions,
            fft_size: n,
            total_i16,
            total_bytes,
        })
    }

    /// Prepares a [`FixedQ15PartitionedConvolver`] by pre-transforming all Q1.15 IR partitions
    /// into BFP spectra using caller-supplied `&'a mut [i16]` memory.
    pub fn prepare(
        ir: &[i16],
        config: FixedQ15PartitionedConfig,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        Self::prepare_from(&ir, config, workspace)
    }

    /// Prepares from Q1.31 taps (`tap / 2^31`, e.g. 24/32-bit PCM or float FIR files).
    ///
    /// Each partition is normalized to the full Q1.15 mantissa before its FFT and keeps its own
    /// block exponent, so low-level IR tails retain ~15 significant bits instead of being
    /// rounded on the absolute Q1.15 grid.
    pub fn prepare_q31(
        ir: &[i32],
        config: FixedQ15PartitionedConfig,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        Self::prepare_from(
            &Q31Taps {
                len: ir.len(),
                tap: |i| ir[i],
            },
            config,
            workspace,
        )
    }

    fn prepare_from<S: TapSource>(
        ir: &S,
        config: FixedQ15PartitionedConfig,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        if ir.len() == 0 {
            return Err(ConvError::EmptyIr);
        }
        if ir.len() > config.max_ir_len {
            return Err(ConvError::IrTooLong {
                ir_len: ir.len(),
                max_ir_len: config.max_ir_len,
            });
        }
        let req = Self::memory_requirements(config)?;
        if workspace.len() < req.total_i16 {
            return Err(ConvError::WorkspaceTooSmall {
                required: req.total_i16,
                provided: workspace.len(),
            });
        }

        let fft_size = FftSize::from_block_size(config.block_size)?;
        let n = fft_size.len();
        let b = config.block_size;
        let bins = fft_size.spectrum_bins();
        let p_total = req.num_partitions;
        let part_bins = p_total * bins;

        let overlap_in_off = 0;
        let ir_re_off = overlap_in_off + n;
        let ir_im_off = ir_re_off + part_bins;
        let fdl_re_off = ir_im_off + part_bins;
        let fdl_im_off = fdl_re_off + part_bins;
        let ir_exp_off = fdl_im_off + part_bins;
        let fdl_exp_off = ir_exp_off + p_total;
        let acc_off = fdl_exp_off + p_total;
        let fft_scratch_off = acc_off + 8 * bins;

        workspace[..req.total_i16].fill(0);
        // Empty FDL slots carry the zero-spectrum exponent so they never set the alignment.
        for e in &mut workspace[fdl_exp_off..fdl_exp_off + p_total] {
            *e = exp_to_word(crate::fixed::fft_q31::ZERO_SPECTRUM_EXP);
        }

        let active_partitions = ir.len().div_ceil(b);
        for p in 0..active_partitions {
            let start = p * b;
            let len = (start + b).min(ir.len()) - start;
            let (head, fft_buf) = workspace.split_at_mut(fft_scratch_off);
            let (before_im, from_im) = head.split_at_mut(ir_im_off);
            let re_slot = &mut before_im[ir_re_off + p * bins..ir_re_off + (p + 1) * bins];
            let im_slot = &mut from_im[p * bins..(p + 1) * bins];
            let e = prepare_partition(
                ir,
                start,
                len,
                n,
                &mut fft_buf[..req.total_i16 - fft_scratch_off],
                re_slot,
                im_slot,
            );
            workspace[ir_exp_off + p] = exp_to_word(e);
        }

        Ok(Self {
            workspace,
            fft_size,
            block_size: b,
            bins,
            num_partitions: p_total,
            active_partitions,
            max_ir_len: config.max_ir_len,
            active_ir_len: ir.len(),
            fdl_head: 0,
            saturation_count: 0,
            overlap_in_off,
            ir_re_off,
            ir_im_off,
            fdl_re_off,
            fdl_im_off,
            ir_exp_off,
            fdl_exp_off,
            acc_off,
            fft_scratch_off,
        })
    }

    /// Processes exactly `block_size` Q1.15 input samples and writes `block_size` output samples.
    pub fn process_block(&mut self, input: &[i16], output: &mut [i16]) -> Result<(), ConvError> {
        let b = self.block_size;
        if input.len() != b || output.len() != b {
            return Err(ConvError::MismatchedBufferLength {
                expected: b,
                actual: input.len(),
            });
        }
        let n = self.fft_size.len();
        let bins = self.bins;
        let p_total = self.num_partitions;

        // 50% overlap-save shift: [prev_B | new_B]
        self.workspace.copy_within(
            self.overlap_in_off + b..self.overlap_in_off + n,
            self.overlap_in_off,
        );
        self.workspace[self.overlap_in_off + b..self.overlap_in_off + n].copy_from_slice(input);

        // Advance FDL ring head and transform the window into slot `fdl_head`.
        self.fdl_head = if self.fdl_head + 1 == p_total {
            0
        } else {
            self.fdl_head + 1
        };
        let slot = self.fdl_head;
        {
            let (head, fft_buf) = self.workspace.split_at_mut(self.fft_scratch_off);
            let (before_re, from_re) = head.split_at_mut(self.fdl_re_off);
            let (re_region, im_region) = from_re.split_at_mut(self.fdl_im_off - self.fdl_re_off);
            let time_in = &before_re[self.overlap_in_off..self.overlap_in_off + n];
            let mut st = Q31FftState::new();
            let mut e = 0;
            for step in 0..q31_fft_steps(n) {
                if let Some(x) =
                    q31_rfft_step(n, step, |i| time_in[i] as i32, -15, fft_buf, &mut st)
                {
                    e = x;
                }
            }
            let s_exp = q31_spectrum_to_q15(
                fft_buf,
                n,
                e,
                &mut re_region[slot * bins..(slot + 1) * bins],
                &mut im_region[slot * bins..(slot + 1) * bins],
            );
            self.workspace[self.fdl_exp_off + slot] = exp_to_word(s_exp);
        }

        // Pass 1: maximum product exponent across all active partitions for exact alignment.
        let mut max_exp: i32 = i32::MIN;
        for p in 0..self.active_partitions {
            let s = (self.fdl_head + p_total - p) % p_total;
            let pe = (self.workspace[self.fdl_exp_off + s] as i32)
                + (self.workspace[self.ir_exp_off + p] as i32);
            max_exp = max_exp.max(pe);
        }
        if max_exp == i32::MIN {
            output.fill(0);
            return Ok(());
        }

        // Pass 2: exponent-aligned exact i64 complex multiply-accumulate with guard bits.
        let guard = cmac_guard_bits(self.active_partitions);
        self.workspace[self.acc_off..self.acc_off + 8 * bins].fill(0);
        for p in 0..self.active_partitions {
            let s = (self.fdl_head + p_total - p) % p_total;
            let pe = (self.workspace[self.fdl_exp_off + s] as i32)
                + (self.workspace[self.ir_exp_off + p] as i32);
            cmac_partition(
                self.workspace,
                self.acc_off,
                bins,
                self.fdl_re_off + s * bins,
                self.fdl_im_off + s * bins,
                self.ir_re_off + p * bins,
                self.ir_im_off + p * bins,
                guard as i32 - (max_exp - pe),
            );
        }

        // Inverse transform in place from i32 mantissas; one RNE rounding per output sample.
        let acc_exp = max_exp - 30 - guard as i32;
        // The 8*bins accumulator words host the 4N+4-word in-place inverse transform.
        let acc = &mut self.workspace[self.acc_off..self.acc_off + 8 * bins];
        let Some(e_x) = acc_i64_to_q31_spectrum_in_place(acc, bins, acc_exp) else {
            output.fill(0);
            return Ok(());
        };
        let mut st = Q31FftState::new();
        let mut e_t = 0;
        for step in 0..q31_fft_steps(n) {
            if let Some(x) = q31_irfft_step(n, step, e_x, acc, &mut st) {
                e_t = x;
            }
        }
        let mut sats = 0u64;
        for (i, y) in output.iter_mut().enumerate() {
            let (v, sat) = scale_rne_sat(
                q31_irfft_sample(acc, n, b + i),
                e_t + 15,
                i16::MIN as i64,
                i16::MAX as i64,
            );
            sats += sat as u64;
            *y = v as i16;
        }
        self.saturation_count = self.saturation_count.saturating_add(sats);
        Ok(())
    }

    /// Deterministically clears FDL rings, overlap history, and saturation counter while
    /// preserving the pre-transformed IR partition spectra.
    pub fn reset(&mut self) {
        let n = self.fft_size.len();
        let part_bins = self.num_partitions * self.bins;
        self.workspace[self.overlap_in_off..self.overlap_in_off + n].fill(0);
        self.workspace[self.fdl_re_off..self.fdl_re_off + part_bins].fill(0);
        self.workspace[self.fdl_im_off..self.fdl_im_off + part_bins].fill(0);
        self.workspace[self.fdl_exp_off..self.fdl_exp_off + self.num_partitions]
            .fill(exp_to_word(crate::fixed::fft_q31::ZERO_SPECTRUM_EXP));
        self.workspace[self.acc_off..self.acc_off + 8 * self.bins].fill(0);
        self.fdl_head = 0;
        self.saturation_count = 0;
    }

    /// Algorithmic latency in samples (`0` for block-aligned overlap-save output).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        0
    }

    /// Cumulative output saturation events observed since `prepare()` or `reset()`.
    #[inline]
    pub const fn saturation_count(&self) -> u64 {
        self.saturation_count
    }

    /// Runtime capabilities metadata.
    #[inline]
    pub const fn capabilities(&self) -> FixedQ15PartitionedCapabilities {
        FixedQ15PartitionedCapabilities {
            block_size: self.block_size,
            fft_size: self.fft_size.len(),
            max_ir_len: self.max_ir_len,
            active_ir_len: self.active_ir_len,
            active_partitions: self.active_partitions,
            latency_samples: 0,
        }
    }
}
