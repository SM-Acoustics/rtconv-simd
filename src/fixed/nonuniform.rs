//! Integrated Fixed-Point Q1.15/BFP Non-Uniform Partitioned Convolver (`FixedQ15NonUniformConvolver`).
//!
//! Shares the exact representation-independent `Planning Plane` ([`NonUniformPlan`],
//! [`NonUniformBudget`]) and `Streaming Plane` ([`NonUniformScheduler`]) from `crate::nonuniform`,
//! paired with the Q1.15/BFP `Numeric Plane` from `crate::fixed`.

use crate::fixed::block_float::bfp_normalize_acc_i64_to_q15;
use crate::fixed::fft_q15::{irfft_q15_bfp, rfft_q15_bfp};
use crate::fixed::q15::{round_sat_q15_from_i64, sat_i16_from_i32};
use crate::nonuniform::memory::stage_delay_ring_len;
use crate::nonuniform::{
    BudgetVerdict, MAX_NONUNIFORM_STAGES, NonUniformBudget, NonUniformPlan, NonUniformScheduler,
    NonUniformStageKind, StageWorkItem,
};
use crate::{ConvError, FftSize};

const MAX_BINS: usize = 129;

/// Explicit memory breakdown for [`FixedQ15NonUniformConvolver`] in `i16` elements and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ15NonUniformMemoryRequirements {
    /// Time-domain state buffers (`i16`).
    pub state_i16: usize,
    /// Frequency-domain BFP spectra + exponents (`i16`).
    pub spectral_i16: usize,
    /// Per-stage causal output delay rings (`i16`).
    pub delay_i16: usize,
    /// Shared FFT time/spectrum scratch (`i16`).
    pub scratch_i16: usize,
    /// Total required `&mut [i16]` elements.
    pub total_i16: usize,
    /// Total required workspace size in bytes (`total_i16 * 2`).
    pub total_bytes: usize,
}

/// Runtime capabilities for [`FixedQ15NonUniformConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ15NonUniformCapabilities {
    /// Number of active NUPC stages.
    pub stage_count: usize,
    /// Maximum configured impulse response length.
    pub max_ir_len: usize,
    /// Active impulse response length.
    pub active_ir_len: usize,
    /// Algorithmic latency in samples (always `0`).
    pub latency_samples: usize,
    /// Admission-control budget verdict.
    pub budget_verdict: BudgetVerdict,
}

#[derive(Clone, Copy, Debug, Default)]
struct FixedStageLayout {
    kind: u8,
    direct_coeffs_off: usize,
    direct_hist_off: usize,
    direct_len: usize,
    direct_active_len: usize,
    direct_head: usize,
    overlap_in_off: usize,
    ir_re_off: usize,
    ir_im_off: usize,
    fdl_re_off: usize,
    fdl_im_off: usize,
    ir_exp_off: usize,
    fdl_exp_off: usize,
    delay_ring_off: usize,
    delay_ring_len: usize,
    fdl_head: usize,
    block_size: usize,
    fft_size: usize,
    bins: usize,
    num_partitions: usize,
    active_partitions: usize,
}

/// Zero-latency, caller-allocated Fixed-Point Q1.15/BFP Non-Uniform Partitioned Convolver.
pub struct FixedQ15NonUniformConvolver<'a> {
    plan: NonUniformPlan,
    stages: [FixedStageLayout; MAX_NONUNIFORM_STAGES],
    stage_count: usize,
    scheduler: NonUniformScheduler,
    workspace: &'a mut [i16],
    acc_re: [[i64; MAX_BINS]; MAX_NONUNIFORM_STAGES],
    acc_im: [[i64; MAX_BINS]; MAX_NONUNIFORM_STAGES],
    acc_max_exp: [i32; MAX_NONUNIFORM_STAGES],
    spec_re_scratch_off: usize,
    spec_im_scratch_off: usize,
    time_scratch_off: usize,
    active_ir_len: usize,
    saturation_count: u64,
    budget_verdict: BudgetVerdict,
}

impl<'a> FixedQ15NonUniformConvolver<'a> {
    /// Computes the exact `&mut [i16]` workspace requirements for a validated [`NonUniformPlan`].
    pub fn memory_requirements(
        plan: &NonUniformPlan,
    ) -> Result<FixedQ15NonUniformMemoryRequirements, ConvError> {
        let mut state_i16 = 0usize;
        let mut spectral_i16 = 0usize;
        let mut delay_i16 = 0usize;
        let mut max_fft = 0usize;
        let mut max_bins = 0usize;

        for st in plan.stages() {
            match st.kind {
                NonUniformStageKind::Direct => {
                    let needed = st
                        .ir_len
                        .checked_mul(2)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    state_i16 = state_i16
                        .checked_add(needed)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                }
                NonUniformStageKind::Fdl => {
                    let fft = FftSize::from_len(st.fft_size)?;
                    let bins = fft.spectrum_bins();
                    let part_bins = st
                        .num_partitions
                        .checked_mul(bins)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    state_i16 = state_i16
                        .checked_add(st.fft_size)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    // ir_re, ir_im, fdl_re, fdl_im + ir_exp(P) + fdl_exp(P)
                    let stage_spec = part_bins
                        .checked_mul(4)
                        .and_then(|v| v.checked_add(st.num_partitions.checked_mul(2)?))
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    spectral_i16 = spectral_i16
                        .checked_add(stage_spec)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let dlen = stage_delay_ring_len(st.ir_offset, st.block_size)?;
                    delay_i16 = delay_i16
                        .checked_add(dlen)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    max_fft = max_fft.max(st.fft_size);
                    max_bins = max_bins.max(bins);
                }
            }
        }

        // Shared scratch: spec_re(max_bins) + spec_im(max_bins) + time_scratch(max_fft)
        let scratch_i16 = max_bins
            .checked_mul(2)
            .and_then(|v| v.checked_add(max_fft))
            .ok_or(ConvError::ArithmeticOverflow)?;
        let total_i16 = state_i16
            .checked_add(spectral_i16)
            .and_then(|v| v.checked_add(delay_i16))
            .and_then(|v| v.checked_add(scratch_i16))
            .ok_or(ConvError::ArithmeticOverflow)?;
        let total_bytes = total_i16
            .checked_mul(core::mem::size_of::<i16>())
            .ok_or(ConvError::ArithmeticOverflow)?;

        Ok(FixedQ15NonUniformMemoryRequirements {
            state_i16,
            spectral_i16,
            delay_i16,
            scratch_i16,
            total_i16,
            total_bytes,
        })
    }

    /// Prepares a [`FixedQ15NonUniformConvolver`] from Q1.15 impulse response `ir`,
    /// validated [`NonUniformPlan`], and caller-supplied `&'a mut [i16]` workspace.
    pub fn prepare(
        ir: &[i16],
        plan: NonUniformPlan,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        Self::prepare_with_budget(ir, plan, None, workspace)
    }

    /// Prepares a [`FixedQ15NonUniformConvolver`] while enforcing an optional [`NonUniformBudget`].
    pub fn prepare_with_budget(
        ir: &[i16],
        plan: NonUniformPlan,
        budget: Option<NonUniformBudget>,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        if ir.is_empty() {
            return Err(ConvError::EmptyIr);
        }
        if ir.len() > plan.max_ir_len() {
            return Err(ConvError::IrTooLong {
                ir_len: ir.len(),
                max_ir_len: plan.max_ir_len(),
            });
        }
        let req = Self::memory_requirements(&plan)?;
        let budget_verdict = match budget {
            Some(b) => {
                let v = plan.check_budget(b, req.total_bytes);
                if v == BudgetVerdict::Exceeded {
                    return Err(ConvError::InvalidNonUniformConfig);
                }
                v
            }
            None => BudgetVerdict::Unverified,
        };
        if workspace.len() < req.total_i16 {
            return Err(ConvError::WorkspaceTooSmall {
                required: req.total_i16,
                provided: workspace.len(),
            });
        }

        let mut stages = [FixedStageLayout::default(); MAX_NONUNIFORM_STAGES];
        let mut cursor = 0usize;
        let mut max_fft = 0usize;
        let mut max_bins = 0usize;

        for (idx, st) in plan.stages().iter().enumerate() {
            match st.kind {
                NonUniformStageKind::Direct => {
                    let coeffs_off = cursor;
                    cursor += st.ir_len;
                    let hist_off = cursor;
                    cursor += st.ir_len;
                    let active = ir.len().saturating_sub(st.ir_offset).min(st.ir_len);
                    stages[idx] = FixedStageLayout {
                        kind: 0,
                        direct_coeffs_off: coeffs_off,
                        direct_hist_off: hist_off,
                        direct_len: st.ir_len,
                        direct_active_len: active,
                        direct_head: 0,
                        ..FixedStageLayout::default()
                    };
                }
                NonUniformStageKind::Fdl => {
                    let fft = FftSize::from_len(st.fft_size)?;
                    let bins = fft.spectrum_bins();
                    let part_bins = st.num_partitions * bins;
                    let dlen = stage_delay_ring_len(st.ir_offset, st.block_size)?;

                    let overlap_in_off = cursor;
                    cursor += st.fft_size;
                    let ir_re_off = cursor;
                    cursor += part_bins;
                    let ir_im_off = cursor;
                    cursor += part_bins;
                    let fdl_re_off = cursor;
                    cursor += part_bins;
                    let fdl_im_off = cursor;
                    cursor += part_bins;
                    let ir_exp_off = cursor;
                    cursor += st.num_partitions;
                    let fdl_exp_off = cursor;
                    cursor += st.num_partitions;
                    let delay_ring_off = cursor;
                    cursor += dlen;

                    let stage_active = ir.len().saturating_sub(st.ir_offset).min(st.ir_len);
                    let active_partitions = if stage_active == 0 {
                        0
                    } else {
                        stage_active.div_ceil(st.block_size)
                    };
                    max_fft = max_fft.max(st.fft_size);
                    max_bins = max_bins.max(bins);

                    stages[idx] = FixedStageLayout {
                        kind: 1,
                        overlap_in_off,
                        ir_re_off,
                        ir_im_off,
                        fdl_re_off,
                        fdl_im_off,
                        ir_exp_off,
                        fdl_exp_off,
                        delay_ring_off,
                        delay_ring_len: dlen,
                        fdl_head: 0,
                        block_size: st.block_size,
                        fft_size: st.fft_size,
                        bins,
                        num_partitions: st.num_partitions,
                        active_partitions,
                        ..FixedStageLayout::default()
                    };
                }
            }
        }

        let spec_re_scratch_off = cursor;
        cursor += max_bins;
        let spec_im_scratch_off = cursor;
        cursor += max_bins;
        let time_scratch_off = cursor;
        cursor += max_fft;
        workspace[..cursor].fill(0);

        // Pre-load direct coefficients and pre-transform all FDL IR partitions into Q15 BFP.
        for (idx, st_plan) in plan.stages().iter().enumerate() {
            let st = &stages[idx];
            match st_plan.kind {
                NonUniformStageKind::Direct => {
                    if st.direct_active_len > 0 {
                        workspace
                            [st.direct_coeffs_off..st.direct_coeffs_off + st.direct_active_len]
                            .copy_from_slice(
                                &ir[st_plan.ir_offset..st_plan.ir_offset + st.direct_active_len],
                            );
                    }
                }
                NonUniformStageKind::Fdl => {
                    let fft = FftSize::from_len(st.fft_size)?;
                    let b = st.block_size;
                    let n = st.fft_size;
                    let bins = st.bins;
                    for p in 0..st.active_partitions {
                        let start = st_plan.ir_offset + p * b;
                        let end = (start + b)
                            .min(st_plan.ir_offset + st_plan.ir_len)
                            .min(ir.len());
                        let copy_len = end.saturating_sub(start);
                        workspace[time_scratch_off..time_scratch_off + n].fill(0);
                        if copy_len > 0 {
                            workspace[time_scratch_off..time_scratch_off + copy_len]
                                .copy_from_slice(&ir[start..start + copy_len]);
                        }
                        let (before_im, from_im) = workspace.split_at_mut(st.ir_im_off);
                        let (im_region, from_time) =
                            from_im.split_at_mut(time_scratch_off - st.ir_im_off);
                        let re_slot =
                            &mut before_im[st.ir_re_off + p * bins..st.ir_re_off + (p + 1) * bins];
                        let im_slot = &mut im_region[p * bins..(p + 1) * bins];
                        let exp = rfft_q15_bfp(fft, &from_time[..n], re_slot, im_slot)?;
                        workspace[st.ir_exp_off + p] = exp as i16;
                    }
                }
            }
        }

        let scheduler = NonUniformScheduler::new(&plan);
        Ok(Self {
            plan,
            stages,
            stage_count: plan.stage_count(),
            scheduler,
            workspace,
            acc_re: [[0i64; MAX_BINS]; MAX_NONUNIFORM_STAGES],
            acc_im: [[0i64; MAX_BINS]; MAX_NONUNIFORM_STAGES],
            acc_max_exp: [i32::MIN; MAX_NONUNIFORM_STAGES],
            spec_re_scratch_off,
            spec_im_scratch_off,
            time_scratch_off,
            active_ir_len: ir.len(),
            saturation_count: 0,
            budget_verdict,
        })
    }

    /// Processes an arbitrary-length block of Q1.15 samples (`input.len() == output.len()`).
    pub fn process_block(&mut self, input: &[i16], output: &mut [i16]) -> Result<(), ConvError> {
        if input.len() != output.len() {
            return Err(ConvError::MismatchedBufferLength {
                expected: input.len(),
                actual: output.len(),
            });
        }
        for (&x, y) in input.iter().zip(output.iter_mut()) {
            *y = self.process_sample(x)?;
        }
        Ok(())
    }

    /// Processes a single Q1.15 sample with zero algorithmic latency.
    pub fn process_sample(&mut self, x: i16) -> Result<i16, ConvError> {
        let mut sum: i32 = 0;

        for idx in 0..self.stage_count {
            if self.stages[idx].kind == 0 {
                let st = &mut self.stages[idx];
                if st.direct_active_len > 0 {
                    let len = st.direct_len;
                    let active = st.direct_active_len;
                    let head = st.direct_head;
                    self.workspace[st.direct_hist_off + head] = x;

                    let first_seg = (head + 1).min(active);
                    let mut acc = 0i64;
                    for k in 0..first_seg {
                        acc += (self.workspace[st.direct_coeffs_off + k] as i64)
                            * (self.workspace[st.direct_hist_off + head - k] as i64);
                    }
                    let rem = active - first_seg;
                    if rem > 0 {
                        let mut h_idx = len - 1;
                        for k in first_seg..active {
                            acc += (self.workspace[st.direct_coeffs_off + k] as i64)
                                * (self.workspace[st.direct_hist_off + h_idx] as i64);
                            h_idx = h_idx.saturating_sub(1);
                        }
                    }
                    st.direct_head = if head + 1 == len { 0 } else { head + 1 };
                    let (y_dir, sat) = round_sat_q15_from_i64(acc);
                    if sat {
                        self.saturation_count = self.saturation_count.saturating_add(1);
                    }
                    sum += y_dir as i32;
                }
            } else {
                let b = self.stages[idx].block_size;
                let fill_before = self.scheduler.stage_state(idx).input_fill;
                self.workspace[self.stages[idx].overlap_in_off + b + fill_before] = x;

                let read_pos = self.scheduler.stage_state(idx).delay_read_pos;
                let ring_idx = read_pos % self.stages[idx].delay_ring_len;
                let y_stage = self.workspace[self.stages[idx].delay_ring_off + ring_idx];
                self.workspace[self.stages[idx].delay_ring_off + ring_idx] = 0;
                self.scheduler.stage_state_mut(idx).delay_read_pos = read_pos + 1;
                sum += y_stage as i32;

                let work = self.scheduler.on_fdl_sample(idx, &self.plan);
                match work {
                    StageWorkItem::None => {}
                    StageWorkItem::StartBlockAndCmac {
                        part_start,
                        part_end,
                        finish_ifft,
                    } => {
                        self.fdl_start_block(idx, part_start, part_end, finish_ifft)?;
                    }
                    StageWorkItem::ContinueCmac {
                        part_start,
                        part_end,
                        finish_ifft,
                    } => {
                        self.fdl_continue_cmac(idx, part_start, part_end, finish_ifft)?;
                    }
                }
            }
        }

        self.scheduler.advance_sample();
        let clamped = sat_i16_from_i32(sum);
        if clamped as i32 != sum {
            self.saturation_count = self.saturation_count.saturating_add(1);
        }
        Ok(clamped)
    }

    fn fdl_start_block(
        &mut self,
        idx: usize,
        part_start: usize,
        part_end: usize,
        finish_ifft: bool,
    ) -> Result<(), ConvError> {
        let st = &mut self.stages[idx];
        let b = st.block_size;
        let n = st.fft_size;
        let bins = st.bins;
        let p_total = st.num_partitions;
        let active_p = st.active_partitions;
        let fft = FftSize::from_len(n)?;

        if active_p == 0 {
            self.workspace.copy_within(
                st.overlap_in_off + b..st.overlap_in_off + n,
                st.overlap_in_off,
            );
            if finish_ifft {
                let wp = self.scheduler.stage_state(idx).delay_write_pos;
                for k in 0..b {
                    let r = (wp + k) % st.delay_ring_len;
                    self.workspace[st.delay_ring_off + r] = 0;
                }
                self.scheduler.stage_state_mut(idx).delay_write_pos = wp + b;
            }
            return Ok(());
        }

        st.fdl_head = if st.fdl_head + 1 == p_total {
            0
        } else {
            st.fdl_head + 1
        };
        let slot = st.fdl_head;

        self.workspace.copy_within(
            st.overlap_in_off..st.overlap_in_off + n,
            self.time_scratch_off,
        );
        self.workspace.copy_within(
            st.overlap_in_off + b..st.overlap_in_off + n,
            st.overlap_in_off,
        );

        let fdl_re_off = st.fdl_re_off;
        let fdl_im_off = st.fdl_im_off;
        let fdl_exp_off = st.fdl_exp_off;
        let ir_exp_off = st.ir_exp_off;
        let time_off = self.time_scratch_off;

        let (before_im, from_im) = self.workspace.split_at_mut(fdl_im_off);
        let (im_region, from_time) = from_im.split_at_mut(time_off - fdl_im_off);
        let x_re_slot = &mut before_im[fdl_re_off + slot * bins..fdl_re_off + (slot + 1) * bins];
        let x_im_slot = &mut im_region[slot * bins..(slot + 1) * bins];
        let x_exp = rfft_q15_bfp(fft, &from_time[..n], x_re_slot, x_im_slot)?;
        self.workspace[fdl_exp_off + slot] = x_exp as i16;

        // Pre-compute max product exponent across all active partitions for this block so
        // spread CMAC steps align identically to burst mode.
        let mut max_exp = i32::MIN;
        for p in 0..active_p {
            let s = (slot + p_total - p) % p_total;
            let pe =
                (self.workspace[fdl_exp_off + s] as i32) + (self.workspace[ir_exp_off + p] as i32);
            if pe > max_exp {
                max_exp = pe;
            }
        }
        self.acc_max_exp[idx] = max_exp;
        self.acc_re[idx][..bins].fill(0);
        self.acc_im[idx][..bins].fill(0);

        self.fdl_continue_cmac(idx, part_start, part_end, finish_ifft)
    }

    fn fdl_continue_cmac(
        &mut self,
        idx: usize,
        part_start: usize,
        part_end: usize,
        finish_ifft: bool,
    ) -> Result<(), ConvError> {
        let st = self.stages[idx];
        let b = st.block_size;
        let n = st.fft_size;
        let bins = st.bins;
        let p_total = st.num_partitions;
        let max_exp = self.acc_max_exp[idx];
        let p_end = part_end.min(st.active_partitions);

        if max_exp != i32::MIN && part_start < p_end {
            for p in part_start..p_end {
                let s = (st.fdl_head + p_total - p) % p_total;
                let pe = (self.workspace[st.fdl_exp_off + s] as i32)
                    + (self.workspace[st.ir_exp_off + p] as i32);
                let delta = ((max_exp - pe) as u32).min(62);

                let xr_base = st.fdl_re_off + s * bins;
                let xi_base = st.fdl_im_off + s * bins;
                let hr_base = st.ir_re_off + p * bins;
                let hi_base = st.ir_im_off + p * bins;

                for k in 0..bins {
                    let xr = self.workspace[xr_base + k] as i64;
                    let xi = self.workspace[xi_base + k] as i64;
                    let hr = self.workspace[hr_base + k] as i64;
                    let hi = self.workspace[hi_base + k] as i64;
                    let pr = xr * hr - xi * hi;
                    let pi = xr * hi + xi * hr;
                    self.acc_re[idx][k] += pr >> delta;
                    self.acc_im[idx][k] += pi >> delta;
                }
            }
        }

        if finish_ifft {
            let wp = self.scheduler.stage_state(idx).delay_write_pos;
            if max_exp == i32::MIN || st.active_partitions == 0 {
                for k in 0..b {
                    let r = (wp + k) % st.delay_ring_len;
                    self.workspace[st.delay_ring_off + r] = 0;
                }
                self.scheduler.stage_state_mut(idx).delay_write_pos = wp + b;
                return Ok(());
            }

            let fft = FftSize::from_len(n)?;
            let base_exp = max_exp.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            let (before_im, from_im) = self.workspace.split_at_mut(self.spec_im_scratch_off);
            let (im_region, from_time) =
                from_im.split_at_mut(self.time_scratch_off - self.spec_im_scratch_off);
            let spec_re = &mut before_im[self.spec_re_scratch_off..self.spec_re_scratch_off + bins];
            let spec_im = &mut im_region[..bins];

            let acc_exp = bfp_normalize_acc_i64_to_q15(
                &self.acc_re[idx][..bins],
                &self.acc_im[idx][..bins],
                base_exp,
                spec_re,
                spec_im,
            );
            let sats = irfft_q15_bfp(fft, spec_re, spec_im, acc_exp, &mut from_time[..n])?;
            self.saturation_count = self.saturation_count.saturating_add(sats);

            for k in 0..b {
                let sample = self.workspace[self.time_scratch_off + b + k];
                let r = (wp + k) % st.delay_ring_len;
                self.workspace[st.delay_ring_off + r] = sample;
            }
            self.scheduler.stage_state_mut(idx).delay_write_pos = wp + b;
        }

        Ok(())
    }

    /// Deterministically resets all delay lines, FDL rings, scheduler cursors, and saturation counter.
    pub fn reset(&mut self) {
        for idx in 0..self.stage_count {
            let st = &mut self.stages[idx];
            if st.kind == 0 {
                self.workspace[st.direct_hist_off..st.direct_hist_off + st.direct_len].fill(0);
                st.direct_head = 0;
            } else {
                let part_bins = st.num_partitions * st.bins;
                self.workspace[st.overlap_in_off..st.overlap_in_off + st.fft_size].fill(0);
                self.workspace[st.fdl_re_off..st.fdl_re_off + part_bins].fill(0);
                self.workspace[st.fdl_im_off..st.fdl_im_off + part_bins].fill(0);
                self.workspace[st.fdl_exp_off..st.fdl_exp_off + st.num_partitions].fill(0);
                self.workspace[st.delay_ring_off..st.delay_ring_off + st.delay_ring_len].fill(0);
                st.fdl_head = 0;
            }
            self.acc_re[idx].fill(0);
            self.acc_im[idx].fill(0);
            self.acc_max_exp[idx] = i32::MIN;
        }
        self.scheduler.reset(&self.plan);
        self.saturation_count = 0;
    }

    /// Algorithmic latency in samples (always `0`).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        0
    }

    /// Total scheduler deadline misses (always `0` for valid causal plans).
    #[inline]
    pub fn deadline_misses(&self) -> u64 {
        self.scheduler.total_deadline_misses()
    }

    /// Cumulative saturation events observed since `prepare()` or `reset()`.
    #[inline]
    pub const fn saturation_count(&self) -> u64 {
        self.saturation_count
    }

    /// Runtime capabilities metadata.
    #[inline]
    pub const fn capabilities(&self) -> FixedQ15NonUniformCapabilities {
        FixedQ15NonUniformCapabilities {
            stage_count: self.stage_count,
            max_ir_len: self.plan.max_ir_len(),
            active_ir_len: self.active_ir_len,
            latency_samples: 0,
            budget_verdict: self.budget_verdict,
        }
    }
}
