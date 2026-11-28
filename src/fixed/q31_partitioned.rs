//! Q1.31 uniform partitioned overlap-save convolver (`FixedQ31PartitionedConvolver`) on the
//! 32-bit-mantissa BFP real FFT core (`crate::fixed::fft_q31`).
//!
//! `#![no_std]`, zero heap, `#![forbid(unsafe_code)]`, integer arithmetic only. All memory is
//! one caller-owned `&mut [i32]` workspace:
//! `overlap(N) + ir_re/ir_im/fdl_re/fdl_im(P·bins each) + ir_exp/fdl_exp(P each)
//! + acc(4·bins: i64 re/im, then the in-place inverse transform) + forward buffer(N + 2)`.
//!
//! The frequency-domain delay line keeps `i32` mantissas with one exponent per partition
//! spectrum, normalized to `|mant| <= 2^b` with `b = min(27, ⌊(62 − bitlen(P))/2⌋)` so that the
//! exponent-aligned products of all `P` partitions accumulate in `i64` without overflow. The
//! only roundings are the transforms (~29-bit mantissas), the spectrum storage (`b + 1` bits),
//! the exponent alignment of products more than `0` bits below the block maximum (round to
//! nearest even) and one final round-to-nearest-even per output sample, with symmetric
//! saturation to `±(2^31 − 1)`.

use crate::fixed::fft_q15::fixed_fft_supports_size;
use crate::fixed::fft_q31::{
    Q31FftState, ZERO_SPECTRUM_EXP, acc_i64_to_q31_spectrum_in_place, q31_fft_steps,
    q31_irfft_sample, q31_irfft_step, q31_rfft_step, q31_spectrum_to_i32, scale_rne_sat,
};
use crate::fixed::spectral::{cmac_partition_q31, prepare_partition_q31, q31_storage_bits};
use crate::{ConvError, FftSize};

/// Configuration for [`FixedQ31PartitionedConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ31PartitionedConfig {
    /// Block size `B` (`32..=1024`, FFT `N = 2B` from `N64` to `N2048`).
    pub block_size: usize,
    /// Maximum supported impulse response length in taps.
    pub max_ir_len: usize,
}

/// Caller-supplied `&mut [i32]` workspace requirements for [`FixedQ31PartitionedConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ31PartitionedMemoryRequirements {
    /// Number of uniform partitions `ceil(max_ir_len / block_size)`.
    pub num_partitions: usize,
    /// FFT length `N = 2 * block_size`.
    pub fft_size: usize,
    /// Total required `&mut [i32]` words.
    pub total_i32: usize,
    /// Total workspace bytes (`total_i32 * 4`).
    pub total_bytes: usize,
}

/// Runtime metadata for [`FixedQ31PartitionedConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ31PartitionedCapabilities {
    /// Block size `B`.
    pub block_size: usize,
    /// FFT length `N = 2B`.
    pub fft_size: usize,
    /// Maximum configured impulse response length.
    pub max_ir_len: usize,
    /// Active impulse response length.
    pub active_ir_len: usize,
    /// Active partition count.
    pub active_partitions: usize,
    /// Spectrum mantissa bound `b` (`|mant| <= 2^b`).
    pub spectrum_bits: u32,
    /// Algorithmic latency in samples (`0`: block-aligned overlap-save output).
    pub latency_samples: usize,
}

/// Uniform partitioned overlap-save FDL convolver in Q1.31 block floating point.
pub struct FixedQ31PartitionedConvolver<'a> {
    workspace: &'a mut [i32],
    n: usize,
    block_size: usize,
    bins: usize,
    num_partitions: usize,
    active_partitions: usize,
    max_ir_len: usize,
    active_ir_len: usize,
    bits: u32,
    fdl_head: usize,
    saturation_count: u64,
    ir_re_off: usize,
    ir_im_off: usize,
    fdl_re_off: usize,
    fdl_im_off: usize,
    ir_exp_off: usize,
    fdl_exp_off: usize,
    acc_off: usize,
    fft_off: usize,
}

impl<'a> FixedQ31PartitionedConvolver<'a> {
    /// Exact `&mut [i32]` workspace requirements for `config`.
    pub fn memory_requirements(
        config: FixedQ31PartitionedConfig,
    ) -> Result<FixedQ31PartitionedMemoryRequirements, ConvError> {
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
        let total_i32 = part_bins
            .checked_mul(4)
            .and_then(|v| v.checked_add(num_partitions.checked_mul(2)?))
            .and_then(|v| v.checked_add(n + 4 * bins + n + 2))
            .ok_or(ConvError::ArithmeticOverflow)?;
        let total_bytes = total_i32
            .checked_mul(4)
            .ok_or(ConvError::ArithmeticOverflow)?;
        Ok(FixedQ31PartitionedMemoryRequirements {
            num_partitions,
            fft_size: n,
            total_i32,
            total_bytes,
        })
    }

    /// Prepares the convolver from Q1.31 taps (`tap / 2^31`).
    pub fn prepare(
        ir: &[i32],
        config: FixedQ31PartitionedConfig,
        workspace: &'a mut [i32],
    ) -> Result<Self, ConvError> {
        Self::prepare_with(ir.len(), |i| ir[i], config, workspace)
    }

    /// Prepares the convolver from `taps` Q1.31 taps produced by `tap(i)`.
    pub fn prepare_with<F: Fn(usize) -> i32>(
        taps: usize,
        tap: F,
        config: FixedQ31PartitionedConfig,
        workspace: &'a mut [i32],
    ) -> Result<Self, ConvError> {
        if taps == 0 {
            return Err(ConvError::EmptyIr);
        }
        if taps > config.max_ir_len {
            return Err(ConvError::IrTooLong {
                ir_len: taps,
                max_ir_len: config.max_ir_len,
            });
        }
        let req = Self::memory_requirements(config)?;
        if workspace.len() < req.total_i32 {
            return Err(ConvError::WorkspaceTooSmall {
                required: req.total_i32,
                provided: workspace.len(),
            });
        }
        let n = req.fft_size;
        let b = config.block_size;
        let bins = n / 2 + 1;
        let p_total = req.num_partitions;
        let part_bins = p_total * bins;
        let ir_re_off = n;
        let ir_im_off = ir_re_off + part_bins;
        let fdl_re_off = ir_im_off + part_bins;
        let fdl_im_off = fdl_re_off + part_bins;
        let ir_exp_off = fdl_im_off + part_bins;
        let fdl_exp_off = ir_exp_off + p_total;
        let acc_off = fdl_exp_off + p_total;
        let fft_off = acc_off + 4 * bins;

        let ws = &mut workspace[..req.total_i32];
        ws.fill(0);
        ws[fdl_exp_off..fdl_exp_off + p_total].fill(ZERO_SPECTRUM_EXP);
        let active_partitions = taps.div_ceil(b);
        let bits = q31_storage_bits(active_partitions);
        for p in 0..active_partitions {
            let start = p * b;
            let len = (start + b).min(taps) - start;
            let (head, fft_buf) = ws.split_at_mut(fft_off);
            let (lo, hi) = head.split_at_mut(ir_im_off);
            let e = prepare_partition_q31(
                &tap,
                start,
                len,
                n,
                bits,
                &mut fft_buf[..n + 2],
                &mut lo[ir_re_off + p * bins..ir_re_off + (p + 1) * bins],
                &mut hi[p * bins..(p + 1) * bins],
            );
            ws[ir_exp_off + p] = e;
        }

        Ok(Self {
            workspace,
            n,
            block_size: b,
            bins,
            num_partitions: p_total,
            active_partitions,
            max_ir_len: config.max_ir_len,
            active_ir_len: taps,
            bits,
            fdl_head: 0,
            saturation_count: 0,
            ir_re_off,
            ir_im_off,
            fdl_re_off,
            fdl_im_off,
            ir_exp_off,
            fdl_exp_off,
            acc_off,
            fft_off,
        })
    }

    /// Processes exactly `block_size` Q1.31 samples.
    pub fn process_block(&mut self, input: &[i32], output: &mut [i32]) -> Result<(), ConvError> {
        let b = self.block_size;
        if input.len() != b || output.len() != b {
            return Err(ConvError::MismatchedBufferLength {
                expected: b,
                actual: input.len(),
            });
        }
        let n = self.n;
        let bins = self.bins;
        let p_total = self.num_partitions;

        // 50% overlap-save window [previous B | new B].
        self.workspace.copy_within(b..n, 0);
        self.workspace[b..n].copy_from_slice(input);

        self.fdl_head = if self.fdl_head + 1 == p_total {
            0
        } else {
            self.fdl_head + 1
        };
        let slot = self.fdl_head;
        {
            let (head, fft_buf) = self.workspace.split_at_mut(self.fft_off);
            let (lo, from_re) = head.split_at_mut(self.fdl_re_off);
            let (re, im) = from_re.split_at_mut(self.fdl_im_off - self.fdl_re_off);
            let time_in = &lo[..n];
            let buf = &mut fft_buf[..n + 2];
            let mut st = Q31FftState::new();
            let mut e = ZERO_SPECTRUM_EXP;
            for step in 0..q31_fft_steps(n) {
                if let Some(x) = q31_rfft_step(n, step, |i| time_in[i], -31, buf, &mut st) {
                    e = x;
                }
            }
            let s_exp = q31_spectrum_to_i32(
                buf,
                n,
                e,
                self.bits,
                &mut re[slot * bins..(slot + 1) * bins],
                &mut im[slot * bins..(slot + 1) * bins],
            );
            self.workspace[self.fdl_exp_off + slot] = s_exp;
        }

        let mut max_exp = i32::MIN;
        for p in 0..self.active_partitions {
            let s = (self.fdl_head + p_total - p) % p_total;
            let pe = self.workspace[self.fdl_exp_off + s] + self.workspace[self.ir_exp_off + p];
            max_exp = max_exp.max(pe);
        }

        self.workspace[self.acc_off..self.acc_off + 4 * bins].fill(0);
        for p in 0..self.active_partitions {
            let s = (self.fdl_head + p_total - p) % p_total;
            let pe = self.workspace[self.fdl_exp_off + s] + self.workspace[self.ir_exp_off + p];
            cmac_partition_q31(
                self.workspace,
                self.acc_off,
                bins,
                self.fdl_re_off + s * bins,
                self.fdl_im_off + s * bins,
                self.ir_re_off + p * bins,
                self.ir_im_off + p * bins,
                (max_exp - pe) as u32,
            );
        }

        let acc = &mut self.workspace[self.acc_off..self.acc_off + 4 * bins];
        let Some(e_x) = acc_i64_to_q31_spectrum_in_place(acc, bins, max_exp) else {
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
                e_t + 31,
                -(i32::MAX as i64),
                i32::MAX as i64,
            );
            sats += sat as u64;
            *y = v as i32;
        }
        self.saturation_count = self.saturation_count.saturating_add(sats);
        Ok(())
    }

    /// Clears the overlap history, the frequency-domain delay line and the saturation counter,
    /// keeping the filter.
    pub fn reset(&mut self) {
        let part_bins = self.num_partitions * self.bins;
        self.workspace[..self.n].fill(0);
        self.workspace[self.fdl_re_off..self.fdl_re_off + 2 * part_bins].fill(0);
        self.workspace[self.fdl_exp_off..self.fdl_exp_off + self.num_partitions]
            .fill(ZERO_SPECTRUM_EXP);
        self.fdl_head = 0;
        self.saturation_count = 0;
    }

    /// Algorithmic latency in samples (`0` for block-aligned overlap-save output).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        0
    }

    /// Output samples clipped by saturation since `prepare` or [`Self::reset`].
    #[inline]
    pub const fn saturation_count(&self) -> u64 {
        self.saturation_count
    }

    /// Runtime metadata.
    pub const fn capabilities(&self) -> FixedQ31PartitionedCapabilities {
        FixedQ31PartitionedCapabilities {
            block_size: self.block_size,
            fft_size: self.n,
            max_ir_len: self.max_ir_len,
            active_ir_len: self.active_ir_len,
            active_partitions: self.active_partitions,
            spectrum_bits: self.bits,
            latency_samples: 0,
        }
    }
}
