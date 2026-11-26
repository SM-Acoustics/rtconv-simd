//! Pure Rust `#![no_std]` Q1.15 Block-Floating-Point uniform partitioned FDL convolver
//! (`FixedQ15PartitionedConvolver`).
//!
//! Zero heap allocation, `#![forbid(unsafe_code)]`, FPU-free integer arithmetic with exact
//! per-partition BFP exponent alignment and saturation telemetry.

use crate::fixed::block_float::{
    Q31Taps, TapSource, bfp_normalize_acc_i16ws_to_q15, load_i64_from_i16, store_i64_to_i16,
};
use crate::fixed::fft_q15::{
    fixed_fft_scratch_len_i16, fixed_fft_supports_size, irfft_q15_bfp_ws, rfft_q15_bfp_ws,
};
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
    acc_re_off: usize,
    acc_im_off: usize,
    acc_spec_re_off: usize,
    acc_spec_im_off: usize,
    time_scratch_off: usize,
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
        let fft_sc = fixed_fft_scratch_len_i16(fft);

        // Layout:
        // overlap_in(N) + ir_re(P*bins) + ir_im(P*bins) + fdl_re(P*bins) + fdl_im(P*bins)
        // + ir_exp(P) + fdl_exp(P) + acc_re(4*bins) + acc_im(4*bins)
        // + acc_spec_re(bins) + acc_spec_im(bins) + time_scratch(N) + fft_scratch(4*N)
        let total_i16 = n
            .checked_mul(2)
            .and_then(|v| v.checked_add(part_bins.checked_mul(4)?))
            .and_then(|v| v.checked_add(num_partitions.checked_mul(2)?))
            .and_then(|v| v.checked_add(bins.checked_mul(10)?))
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
        let fft_sc = fixed_fft_scratch_len_i16(fft_size);

        let overlap_in_off = 0;
        let ir_re_off = overlap_in_off + n;
        let ir_im_off = ir_re_off + part_bins;
        let fdl_re_off = ir_im_off + part_bins;
        let fdl_im_off = fdl_re_off + part_bins;
        let ir_exp_off = fdl_im_off + part_bins;
        let fdl_exp_off = ir_exp_off + p_total;
        let acc_re_off = fdl_exp_off + p_total;
        let acc_im_off = acc_re_off + 4 * bins;
        let acc_spec_re_off = acc_im_off + 4 * bins;
        let acc_spec_im_off = acc_spec_re_off + bins;
        let time_scratch_off = acc_spec_im_off + bins;
        let fft_scratch_off = time_scratch_off + n;

        workspace[..req.total_i16].fill(0);

        let active_partitions = ir.len().div_ceil(b);
        for p in 0..active_partitions {
            let start = p * b;
            let end = (start + b).min(ir.len());
            let copy_len = end - start;
            let exp_offset = ir.fill_block(
                start,
                copy_len,
                &mut workspace[time_scratch_off..time_scratch_off + n],
            );

            let (before_im, from_im) = workspace.split_at_mut(ir_im_off);
            let (im_region, from_time) = from_im.split_at_mut(time_scratch_off - ir_im_off);
            let (time_region, fft_sc_region) =
                from_time.split_at_mut(fft_scratch_off - time_scratch_off);
            let re_slot = &mut before_im[ir_re_off + p * bins..ir_re_off + (p + 1) * bins];
            let im_slot = &mut im_region[p * bins..(p + 1) * bins];
            let exp = rfft_q15_bfp_ws(
                fft_size,
                &time_region[..n],
                re_slot,
                im_slot,
                &mut fft_sc_region[..fft_sc],
            )?;
            workspace[ir_exp_off + p] = (exp as i32 + exp_offset) as i16;
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
            acc_re_off,
            acc_im_off,
            acc_spec_re_off,
            acc_spec_im_off,
            time_scratch_off,
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
        let fft_sc = fixed_fft_scratch_len_i16(self.fft_size);

        // 50% overlap-save shift: [prev_B | new_B]
        self.workspace.copy_within(
            self.overlap_in_off + b..self.overlap_in_off + n,
            self.overlap_in_off,
        );
        self.workspace[self.overlap_in_off + b..self.overlap_in_off + n].copy_from_slice(input);

        // Advance FDL ring head and RFFT into slot `fdl_head`.
        self.fdl_head = if self.fdl_head + 1 == p_total {
            0
        } else {
            self.fdl_head + 1
        };

        let slot = self.fdl_head;
        let (before_fdl_re, from_fdl_re) = self.workspace.split_at_mut(self.fdl_re_off);
        let (fdl_re_region, from_fdl_im) =
            from_fdl_re.split_at_mut(self.fdl_im_off - self.fdl_re_off);
        let (fdl_im_region, fft_sc_region) =
            from_fdl_im.split_at_mut(self.fft_scratch_off - self.fdl_im_off);
        let time_in = &before_fdl_re[self.overlap_in_off..self.overlap_in_off + n];
        let x_re_slot = &mut fdl_re_region[slot * bins..(slot + 1) * bins];
        let x_im_slot = &mut fdl_im_region[slot * bins..(slot + 1) * bins];
        let x_exp = rfft_q15_bfp_ws(
            self.fft_size,
            time_in,
            x_re_slot,
            x_im_slot,
            &mut fft_sc_region[..fft_sc],
        )?;
        self.workspace[self.fdl_exp_off + slot] = x_exp as i16;

        // Pass 1: find maximum product exponent across all active partitions for exact alignment.
        let mut max_exp: i32 = i32::MIN;
        for p in 0..self.active_partitions {
            let s = (self.fdl_head + p_total - p) % p_total;
            let pe = (self.workspace[self.fdl_exp_off + s] as i32)
                + (self.workspace[self.ir_exp_off + p] as i32);
            if pe > max_exp {
                max_exp = pe;
            }
        }
        if max_exp == i32::MIN {
            output.fill(0);
            return Ok(());
        }

        // Pass 2: exponent-aligned i64 complex multiply-accumulate in caller workspace.
        self.workspace[self.acc_re_off..self.acc_re_off + 4 * bins].fill(0);
        self.workspace[self.acc_im_off..self.acc_im_off + 4 * bins].fill(0);
        for p in 0..self.active_partitions {
            let s = (self.fdl_head + p_total - p) % p_total;
            let pe = (self.workspace[self.fdl_exp_off + s] as i32)
                + (self.workspace[self.ir_exp_off + p] as i32);
            let delta = ((max_exp - pe) as u32).min(62);

            let xr_base = self.fdl_re_off + s * bins;
            let xi_base = self.fdl_im_off + s * bins;
            let hr_base = self.ir_re_off + p * bins;
            let hi_base = self.ir_im_off + p * bins;

            for k in 0..bins {
                let xr = self.workspace[xr_base + k] as i64;
                let xi = self.workspace[xi_base + k] as i64;
                let hr = self.workspace[hr_base + k] as i64;
                let hi = self.workspace[hi_base + k] as i64;
                let pr = xr * hr - xi * hi;
                let pi = xr * hi + xi * hr;
                let re_off = self.acc_re_off + k * 4;
                let im_off = self.acc_im_off + k * 4;
                let ar = load_i64_from_i16(self.workspace, re_off) + (pr >> delta);
                let ai = load_i64_from_i16(self.workspace, im_off) + (pi >> delta);
                store_i64_to_i16(self.workspace, re_off, ar);
                store_i64_to_i16(self.workspace, im_off, ai);
            }
        }

        // Normalize i64 Q2.30 accumulator to Q1.15 BFP spectrum and run IRFFT in caller scratch.
        let base_exp = max_exp.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        let (before_spec_re, from_spec_re) = self.workspace.split_at_mut(self.acc_spec_re_off);
        let (spec_re_region, from_spec_im) =
            from_spec_re.split_at_mut(self.acc_spec_im_off - self.acc_spec_re_off);
        let (spec_im_region, from_time) =
            from_spec_im.split_at_mut(self.time_scratch_off - self.acc_spec_im_off);
        let (time_region, fft_sc_region) =
            from_time.split_at_mut(self.fft_scratch_off - self.time_scratch_off);
        let acc_re_ws = &before_spec_re[self.acc_re_off..self.acc_re_off + 4 * bins];
        let acc_im_ws = &before_spec_re[self.acc_im_off..self.acc_im_off + 4 * bins];
        let spec_re = &mut spec_re_region[..bins];
        let spec_im = &mut spec_im_region[..bins];

        let acc_exp =
            bfp_normalize_acc_i16ws_to_q15(acc_re_ws, acc_im_ws, bins, base_exp, spec_re, spec_im);
        let sats = irfft_q15_bfp_ws(
            self.fft_size,
            spec_re,
            spec_im,
            acc_exp,
            &mut time_region[..n],
            &mut fft_sc_region[..fft_sc],
        )?;
        self.saturation_count = self.saturation_count.saturating_add(sats);

        // Extract valid 50% overlap-save second half (`time_scratch[B..2*B]`).
        output.copy_from_slice(&time_region[b..n]);
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
        self.workspace[self.fdl_exp_off..self.fdl_exp_off + self.num_partitions].fill(0);
        self.workspace[self.acc_re_off..self.acc_re_off + 4 * self.bins].fill(0);
        self.workspace[self.acc_im_off..self.acc_im_off + 4 * self.bins].fill(0);
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
