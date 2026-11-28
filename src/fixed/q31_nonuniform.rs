//! Integrated Q1.31 Non-Uniform Partitioned Convolver (`FixedQ31NonUniformConvolver`) for
//! 24/32-bit audio on FPU-less MCUs.
//!
//! Shares the representation-independent planning plane ([`NonUniformPlan`],
//! [`NonUniformBudget`], `NonUniformPlan::optimize`) and streaming plane
//! ([`NonUniformScheduler`]: time-distributed transform steps, CMAC slices, clearance deferral,
//! zero algorithmic latency) with the `f32` and Q1.15 engines, paired with a Q1.31 numeric
//! plane:
//! - the direct head multiplies Q1.31 samples by Q1.31 taps exactly in `i64` (product
//!   pre-shift only when `‖h_head‖₁ >= 2`, as in `crate::fixed::q31_direct`);
//! - FDL stages transform with the 32-bit-mantissa BFP real FFT core (`crate::fixed::fft_q31`)
//!   and keep the spectra as `i32` mantissas with one exponent per partition, bounded
//!   (`|mant| <= 2^b`, `b = min(27, ⌊(62 − bitlen(P))/2⌋)`) so the aligned products of all `P`
//!   partitions accumulate in `i64` without overflow;
//! - stage outputs land in power-of-two `2·B_s` delay rings as `i64` with 47 fractional bits;
//!   head and rings are summed and rounded to Q1.31 **once** per sample (round to nearest
//!   even, symmetric saturation to `±(2^31 − 1)`).
//!
//! All buffers live in one caller-owned `&mut [i32]` workspace; the value itself holds only
//! stage metadata and scheduler state.

use crate::fixed::block_float::round_shift_even_i64;
use crate::fixed::fft_q15::fixed_fft_supports_size;
use crate::fixed::fft_q31::{
    Q31FftState, ZERO_SPECTRUM_EXP, acc_i64_to_q31_spectrum_in_place, q31_fft_steps,
    q31_irfft_sample, q31_irfft_step, q31_rfft_step, q31_spectrum_to_i32, scale_rne_sat,
};
use crate::fixed::q31_direct::{q31_product_shift_with, round_sat_q31_sym};
use crate::fixed::spectral::{cmac_partition_q31, prepare_partition_q31, q31_storage_bits};
use crate::nonuniform::memory::{shared_input_ring_len, stage_delay_ring_len};
use crate::nonuniform::{
    BudgetVerdict, MAX_NONUNIFORM_STAGES, NonUniformBudget, NonUniformPlan, NonUniformScheduler,
    NonUniformStageKind, StageWorkItem,
};
use crate::{ConvError, FftSize};

/// Fractional bits of the `i64` stage-output rings (`Q16.47`): 16 bits below the Q1.31 LSB so
/// head and stages are rounded to the output grid exactly once.
const RING_FRAC_BITS: u32 = 47;

#[inline(always)]
fn ring_load(ws: &[i32], off: usize) -> i64 {
    ((ws[off] as i64) << 32) | (ws[off + 1] as u32 as i64)
}

#[inline(always)]
fn ring_store(ws: &mut [i32], off: usize, v: i64) {
    ws[off] = (v >> 32) as i32;
    ws[off + 1] = v as i32;
}

/// Explicit memory breakdown for [`FixedQ31NonUniformConvolver`] in `i32` words and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ31NonUniformMemoryRequirements {
    /// Time-domain state (direct-head coefficients + history, plus the single shared Q1.31
    /// input ring of all FDL stages).
    pub state_i32: usize,
    /// `i32` BFP spectra, exponents and per-stage `i64` accumulators (`4 * (B_s + 1)` words per
    /// stage, which also host the in-place staged FFT (`N_s + 2`) and IFFT (`2 * N_s + 2`)).
    pub spectral_i32: usize,
    /// Power-of-two causal output delay rings (`2 * B_s` `i64` samples = `4 * B_s` words per
    /// FDL stage).
    pub delay_i32: usize,
    /// Shared overlap-save time window of the largest FDL stage (`max N_s` words).
    pub scratch_i32: usize,
    /// Total required `&mut [i32]` words.
    pub total_i32: usize,
    /// Total workspace bytes (`total_i32 * 4`).
    pub total_bytes: usize,
    /// Size in bytes of the [`FixedQ31NonUniformConvolver`] value itself (included in budget
    /// admission checks).
    pub instance_bytes: usize,
}

/// Runtime capabilities for [`FixedQ31NonUniformConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ31NonUniformCapabilities {
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
struct Q31StageLayout {
    kind: u8,
    direct_coeffs_off: usize,
    direct_hist_off: usize,
    direct_len: usize,
    direct_active_len: usize,
    direct_head: usize,
    direct_shift: u32,
    ir_re_off: usize,
    ir_im_off: usize,
    fdl_re_off: usize,
    fdl_im_off: usize,
    ir_exp_off: usize,
    fdl_exp_off: usize,
    acc_off: usize,
    delay_ring_off: usize,
    delay_ring_len: usize,
    delay_ring_mask: usize,
    defer_samples: usize,
    fft_steps: usize,
    fdl_head: usize,
    block_size: usize,
    fft_size: usize,
    bins: usize,
    bits: u32,
    num_partitions: usize,
    active_partitions: usize,
}

/// Zero-latency, caller-allocated Q1.31 non-uniform partitioned convolver.
///
/// FDL stage FFT sizes `N64` through `N2048` (`block_size` `32..=1024`); plans with larger
/// stages return [`ConvError::UnsupportedFftSize`].
pub struct FixedQ31NonUniformConvolver<'a> {
    plan: NonUniformPlan,
    shared_in_off: usize,
    shared_in_len: usize,
    shared_in_write_pos: usize,
    stages: [Q31StageLayout; MAX_NONUNIFORM_STAGES],
    stage_count: usize,
    scheduler: NonUniformScheduler,
    workspace: &'a mut [i32],
    acc_max_exp: [i32; MAX_NONUNIFORM_STAGES],
    rfft_state: [Q31FftState; MAX_NONUNIFORM_STAGES],
    ifft_state: [Q31FftState; MAX_NONUNIFORM_STAGES],
    time_scratch_off: usize,
    active_ir_len: usize,
    saturation_count: u64,
    budget_verdict: BudgetVerdict,
}

impl<'a> FixedQ31NonUniformConvolver<'a> {
    /// Exact `&mut [i32]` workspace requirements for a validated [`NonUniformPlan`].
    pub fn memory_requirements(
        plan: &NonUniformPlan,
    ) -> Result<FixedQ31NonUniformMemoryRequirements, ConvError> {
        let ovf = ConvError::ArithmeticOverflow;
        let mut state = shared_input_ring_len(plan)?;
        let mut spectral = 0usize;
        let mut delay = 0usize;
        let mut max_fft = 0usize;
        for st in plan.stages() {
            match st.kind {
                NonUniformStageKind::Direct => {
                    state = state
                        .checked_add(st.ir_len.checked_mul(2).ok_or(ovf)?)
                        .ok_or(ovf)?;
                }
                NonUniformStageKind::Fdl => {
                    let fft = FftSize::from_len(st.fft_size)?;
                    if !fixed_fft_supports_size(fft) {
                        return Err(ConvError::UnsupportedFftSize {
                            fft_size: st.fft_size,
                        });
                    }
                    let bins = fft.spectrum_bins();
                    let part_bins = st.num_partitions.checked_mul(bins).ok_or(ovf)?;
                    // ir_re, ir_im, fdl_re, fdl_im + ir_exp(P) + fdl_exp(P) + acc(4*bins)
                    let stage_spec = part_bins
                        .checked_mul(4)
                        .and_then(|v| v.checked_add(st.num_partitions.checked_mul(2)?))
                        .and_then(|v| v.checked_add(bins.checked_mul(4)?))
                        .ok_or(ovf)?;
                    spectral = spectral.checked_add(stage_spec).ok_or(ovf)?;
                    let dlen = stage_delay_ring_len(st.block_size)?;
                    delay = delay
                        .checked_add(dlen.checked_mul(2).ok_or(ovf)?)
                        .ok_or(ovf)?;
                    max_fft = max_fft.max(st.fft_size);
                }
            }
        }
        let total_i32 = state
            .checked_add(spectral)
            .and_then(|v| v.checked_add(delay))
            .and_then(|v| v.checked_add(max_fft))
            .ok_or(ovf)?;
        Ok(FixedQ31NonUniformMemoryRequirements {
            state_i32: state,
            spectral_i32: spectral,
            delay_i32: delay,
            scratch_i32: max_fft,
            total_i32,
            total_bytes: total_i32.checked_mul(4).ok_or(ovf)?,
            instance_bytes: core::mem::size_of::<FixedQ31NonUniformConvolver<'static>>(),
        })
    }

    /// Prepares the convolver from Q1.31 taps `ir` (`tap / 2^31`).
    pub fn prepare(
        ir: &[i32],
        plan: NonUniformPlan,
        workspace: &'a mut [i32],
    ) -> Result<Self, ConvError> {
        Self::prepare_with(ir.len(), |i| ir[i], plan, None, workspace)
    }

    /// Prepares while enforcing an optional [`NonUniformBudget`] (memory includes
    /// `instance_bytes`).
    pub fn prepare_with_budget(
        ir: &[i32],
        plan: NonUniformPlan,
        budget: Option<NonUniformBudget>,
        workspace: &'a mut [i32],
    ) -> Result<Self, ConvError> {
        Self::prepare_with(ir.len(), |i| ir[i], plan, budget, workspace)
    }

    /// Prepares from `len` Q1.31 taps supplied by index (`tap(i)`), e.g. decoded on the fly
    /// from a FIR artifact.
    pub fn prepare_with<F: Fn(usize) -> i32>(
        len: usize,
        tap: F,
        plan: NonUniformPlan,
        budget: Option<NonUniformBudget>,
        workspace: &'a mut [i32],
    ) -> Result<Self, ConvError> {
        if len == 0 {
            return Err(ConvError::EmptyIr);
        }
        if len > plan.max_ir_len() {
            return Err(ConvError::IrTooLong {
                ir_len: len,
                max_ir_len: plan.max_ir_len(),
            });
        }
        let req = Self::memory_requirements(&plan)?;
        let budget_verdict = match budget {
            Some(b) => {
                let v = plan.check_budget(b, req.total_bytes.saturating_add(req.instance_bytes));
                if v == BudgetVerdict::Exceeded {
                    return Err(ConvError::InvalidNonUniformConfig);
                }
                v
            }
            None => BudgetVerdict::Unverified,
        };
        if workspace.len() < req.total_i32 {
            return Err(ConvError::WorkspaceTooSmall {
                required: req.total_i32,
                provided: workspace.len(),
            });
        }

        let shared_in_off = 0usize;
        let shared_in_len = shared_input_ring_len(&plan)?;
        let mut stages = [Q31StageLayout::default(); MAX_NONUNIFORM_STAGES];
        let mut cursor = shared_in_len;
        let mut max_fft = 0usize;
        for (idx, st) in plan.stages().iter().enumerate() {
            match st.kind {
                NonUniformStageKind::Direct => {
                    let coeffs_off = cursor;
                    cursor += st.ir_len;
                    let hist_off = cursor;
                    cursor += st.ir_len;
                    let active = len.saturating_sub(st.ir_offset).min(st.ir_len);
                    let shift = q31_product_shift_with(active, |k| tap(st.ir_offset + k));
                    stages[idx] = Q31StageLayout {
                        kind: 0,
                        direct_coeffs_off: coeffs_off,
                        direct_hist_off: hist_off,
                        direct_len: st.ir_len,
                        direct_active_len: active,
                        direct_shift: shift,
                        ..Q31StageLayout::default()
                    };
                }
                NonUniformStageKind::Fdl => {
                    let bins = st.fft_size / 2 + 1;
                    let part_bins = st.num_partitions * bins;
                    let dlen = stage_delay_ring_len(st.block_size)?;
                    let stage_active = len.saturating_sub(st.ir_offset).min(st.ir_len);
                    let active_partitions = stage_active.div_ceil(st.block_size);
                    let ir_re_off = cursor;
                    let ir_im_off = ir_re_off + part_bins;
                    let fdl_re_off = ir_im_off + part_bins;
                    let fdl_im_off = fdl_re_off + part_bins;
                    let ir_exp_off = fdl_im_off + part_bins;
                    let fdl_exp_off = ir_exp_off + st.num_partitions;
                    let acc_off = fdl_exp_off + st.num_partitions;
                    let delay_ring_off = acc_off + 4 * bins;
                    cursor = delay_ring_off + 2 * dlen;
                    max_fft = max_fft.max(st.fft_size);
                    stages[idx] = Q31StageLayout {
                        kind: 1,
                        ir_re_off,
                        ir_im_off,
                        fdl_re_off,
                        fdl_im_off,
                        ir_exp_off,
                        fdl_exp_off,
                        acc_off,
                        delay_ring_off,
                        delay_ring_len: dlen,
                        delay_ring_mask: dlen - 1,
                        defer_samples: st
                            .slack_samples
                            .saturating_sub(st.block_size.saturating_sub(1)),
                        fft_steps: q31_fft_steps(st.fft_size),
                        block_size: st.block_size,
                        fft_size: st.fft_size,
                        bins,
                        bits: q31_storage_bits(active_partitions.max(1)),
                        num_partitions: st.num_partitions,
                        active_partitions,
                        ..Q31StageLayout::default()
                    };
                }
            }
        }
        let time_scratch_off = cursor;
        cursor += max_fft;
        workspace[..cursor].fill(0);

        // Direct coefficients and FDL partition spectra (each stage's accumulator region is
        // the forward-transform buffer during preparation).
        for (idx, st_plan) in plan.stages().iter().enumerate() {
            let st = stages[idx];
            if st.kind == 0 {
                for k in 0..st.direct_active_len {
                    workspace[st.direct_coeffs_off + k] = tap(st_plan.ir_offset + k);
                }
                continue;
            }
            let (b, n, bins) = (st.block_size, st.fft_size, st.bins);
            for p in 0..st.active_partitions {
                let start = st_plan.ir_offset + p * b;
                let end = (start + b).min(st_plan.ir_offset + st_plan.ir_len).min(len);
                let (lo, from_im) = workspace.split_at_mut(st.ir_im_off);
                let (im_region, from_acc) = from_im.split_at_mut(st.acc_off - st.ir_im_off);
                let e = prepare_partition_q31(
                    &tap,
                    start,
                    end.saturating_sub(start),
                    n,
                    st.bits,
                    &mut from_acc[..4 * bins],
                    &mut lo[st.ir_re_off + p * bins..st.ir_re_off + (p + 1) * bins],
                    &mut im_region[p * bins..(p + 1) * bins],
                );
                workspace[st.ir_exp_off + p] = e;
            }
            workspace[st.acc_off..st.acc_off + 4 * bins].fill(0);
            workspace[st.fdl_exp_off..st.fdl_exp_off + st.num_partitions].fill(ZERO_SPECTRUM_EXP);
        }

        let scheduler = NonUniformScheduler::new(&plan);
        Ok(Self {
            plan,
            shared_in_off,
            shared_in_len,
            shared_in_write_pos: 0,
            stages,
            stage_count: plan.stage_count(),
            scheduler,
            workspace,
            acc_max_exp: [i32::MIN; MAX_NONUNIFORM_STAGES],
            rfft_state: [Q31FftState::new(); MAX_NONUNIFORM_STAGES],
            ifft_state: [Q31FftState::new(); MAX_NONUNIFORM_STAGES],
            time_scratch_off,
            active_ir_len: len,
            saturation_count: 0,
            budget_verdict,
        })
    }

    /// Processes an arbitrary-length block of Q1.31 samples (`input.len() == output.len()`).
    #[inline]
    pub fn process_block(&mut self, input: &[i32], output: &mut [i32]) -> Result<(), ConvError> {
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

    /// Direct-head contribution in ring units (`2^-47`), exact `i64` MAC over the two
    /// contiguous history segments (no per-tap branch).
    #[inline]
    fn direct_stage(&mut self, idx: usize, x: i32) -> i64 {
        let st = &mut self.stages[idx];
        if st.direct_active_len == 0 {
            return 0;
        }
        let (len, active, head) = (st.direct_len, st.direct_active_len, st.direct_head);
        let (c_off, h_off, g) = (st.direct_coeffs_off, st.direct_hist_off, st.direct_shift);
        st.direct_head = if head + 1 == len { 0 } else { head + 1 };
        self.workspace[h_off + head] = x;
        let coeffs = &self.workspace[c_off..c_off + active];
        let hist = &self.workspace[h_off..h_off + len];
        // window[k] = x[n-k]: hist[head], hist[head-1], ..., hist[0], hist[len-1], ...
        let first = (head + 1).min(active);
        let mut acc = 0i64;
        for (&c, &v) in coeffs[..first].iter().zip(hist[..=head].iter().rev()) {
            acc += (c as i64 * v as i64) >> g;
        }
        for (&c, &v) in coeffs[first..].iter().zip(hist[head + 1..].iter().rev()) {
            acc += (c as i64 * v as i64) >> g;
        }
        // acc·2^(g − 62) -> units of 2^-47
        let sh = 62 - g as i32 - RING_FRAC_BITS as i32;
        if sh > 0 {
            round_shift_even_i64(acc, sh as u32)
        } else {
            acc << (-sh) as u32
        }
    }

    /// Processes one Q1.31 sample with zero algorithmic latency.
    pub fn process_sample(&mut self, x: i32) -> Result<i32, ConvError> {
        let in_write_pos = self.shared_in_write_pos;
        if self.shared_in_len > 0 {
            self.workspace[self.shared_in_off + in_write_pos] = x;
        }
        let mut sum: i64 = 0;
        for idx in 0..self.stage_count {
            if self.stages[idx].kind == 0 {
                sum = sum.saturating_add(self.direct_stage(idx, x));
                continue;
            }
            let read_pos = self.scheduler.stage_state(idx).delay_read_pos;
            let off =
                self.stages[idx].delay_ring_off + 2 * (read_pos & self.stages[idx].delay_ring_mask);
            sum = sum.saturating_add(ring_load(self.workspace, off));
            ring_store(self.workspace, off, 0);
            self.scheduler.stage_state_mut(idx).delay_read_pos = read_pos + 1;

            let steps = self.stages[idx].fft_steps;
            match self
                .scheduler
                .on_fdl_sample_with_steps(idx, &self.plan, steps, steps)
            {
                StageWorkItem::None => {}
                StageWorkItem::RfftSteps {
                    step_start,
                    step_end,
                    finish_rfft,
                } => self.fdl_run_rfft_steps(idx, in_write_pos, step_start, step_end, finish_rfft),
                StageWorkItem::ContinueCmac {
                    part_start,
                    part_end,
                    finish_ifft,
                } => {
                    self.fdl_run_cmac(idx, part_start, part_end);
                    if finish_ifft {
                        self.fdl_run_ifft_steps(idx, 0, steps, true);
                    }
                }
                StageWorkItem::IfftSteps {
                    step_start,
                    step_end,
                    commit_output,
                } => self.fdl_run_ifft_steps(idx, step_start, step_end, commit_output),
                StageWorkItem::CombinedSlice {
                    rfft_start,
                    rfft_end,
                    finish_rfft,
                    cmac_start,
                    cmac_end,
                    ifft_start,
                    ifft_end,
                    commit_output,
                } => {
                    if rfft_start < rfft_end {
                        self.fdl_run_rfft_steps(
                            idx,
                            in_write_pos,
                            rfft_start,
                            rfft_end,
                            finish_rfft,
                        );
                    }
                    if cmac_start < cmac_end {
                        self.fdl_run_cmac(idx, cmac_start, cmac_end);
                    }
                    if ifft_start < ifft_end {
                        self.fdl_run_ifft_steps(idx, ifft_start, ifft_end, commit_output);
                    }
                }
            }
        }
        if self.shared_in_len > 0 {
            let next = in_write_pos + 1;
            self.shared_in_write_pos = if next == self.shared_in_len { 0 } else { next };
        }
        self.scheduler.advance_sample();
        // One round-to-nearest-even from 2^-47 units to Q1.31, symmetric saturation.
        let (y, sat) = round_sat_q31_sym(sum, RING_FRAC_BITS - 31);
        self.saturation_count += sat as u64;
        Ok(y)
    }

    #[inline]
    fn copy_shared_input_window(&mut self, in_write_pos: usize, defer_samples: usize, n: usize) {
        let len = self.shared_in_len;
        let off = self.shared_in_off;
        let window_age = (n - 1) + defer_samples;
        let start = if in_write_pos >= window_age {
            in_write_pos - window_age
        } else {
            in_write_pos + len - window_age
        };
        let first = (len - start).min(n);
        self.workspace
            .copy_within(off + start..off + start + first, self.time_scratch_off);
        if first < n {
            self.workspace
                .copy_within(off..off + (n - first), self.time_scratch_off + first);
        }
    }

    fn fdl_run_rfft_steps(
        &mut self,
        idx: usize,
        in_write_pos: usize,
        step_start: usize,
        step_end: usize,
        finish_rfft: bool,
    ) {
        let active_p = self.stages[idx].active_partitions;
        if active_p == 0 || step_start >= step_end {
            return;
        }
        let n = self.stages[idx].fft_size;
        let bins = self.stages[idx].bins;
        let p_total = self.stages[idx].num_partitions;
        if step_start == 0 {
            self.stages[idx].fdl_head = if self.stages[idx].fdl_head + 1 == p_total {
                0
            } else {
                self.stages[idx].fdl_head + 1
            };
            let defer = self.stages[idx].defer_samples;
            self.copy_shared_input_window(in_write_pos, defer, n);
        }
        let slot = self.stages[idx].fdl_head;
        let re_off = self.stages[idx].fdl_re_off + slot * bins;
        let im_off = self.stages[idx].fdl_im_off + slot * bins;
        let acc_off = self.stages[idx].acc_off;
        let fdl_exp_off = self.stages[idx].fdl_exp_off;
        let bits = self.stages[idx].bits;

        let mut finished = None;
        for step in step_start..step_end {
            // Disjoint borrows: spectra slots < acc (in-place buffer) < time window.
            let (head, from_acc) = self.workspace.split_at_mut(acc_off);
            let (acc, from_time) = from_acc.split_at_mut(self.time_scratch_off - acc_off);
            let time_in = &from_time[..n];
            let buf = &mut acc[..4 * bins];
            if let Some(e) =
                q31_rfft_step(n, step, |i| time_in[i], -31, buf, &mut self.rfft_state[idx])
            {
                let (lo, hi) = head.split_at_mut(im_off);
                finished = Some(q31_spectrum_to_i32(
                    buf,
                    n,
                    e,
                    bits,
                    &mut lo[re_off..re_off + bins],
                    &mut hi[..bins],
                ));
            }
        }

        if finish_rfft {
            if let Some(e) = finished {
                self.workspace[fdl_exp_off + slot] = e;
            }
            let ir_exp_off = self.stages[idx].ir_exp_off;
            let mut max_exp = i32::MIN;
            for p in 0..active_p {
                let s = if slot >= p {
                    slot - p
                } else {
                    slot + p_total - p
                };
                max_exp =
                    max_exp.max(self.workspace[fdl_exp_off + s] + self.workspace[ir_exp_off + p]);
            }
            self.acc_max_exp[idx] = max_exp;
            self.workspace[acc_off..acc_off + 4 * bins].fill(0);
        }
    }

    fn fdl_run_cmac(&mut self, idx: usize, part_start: usize, part_end: usize) {
        let st = &self.stages[idx];
        let (bins, p_total, slot) = (st.bins, st.num_partitions, st.fdl_head);
        let max_exp = self.acc_max_exp[idx];
        let p_end = part_end.min(st.active_partitions);
        if max_exp == i32::MIN || part_start >= p_end {
            return;
        }
        let (acc_off, fdl_exp_off, ir_exp_off) = (st.acc_off, st.fdl_exp_off, st.ir_exp_off);
        let (fdl_re, fdl_im, ir_re, ir_im) =
            (st.fdl_re_off, st.fdl_im_off, st.ir_re_off, st.ir_im_off);
        for p in part_start..p_end {
            let s = if slot >= p {
                slot - p
            } else {
                slot + p_total - p
            };
            let pe = self.workspace[fdl_exp_off + s] + self.workspace[ir_exp_off + p];
            cmac_partition_q31(
                self.workspace,
                acc_off,
                bins,
                fdl_re + s * bins,
                fdl_im + s * bins,
                ir_re + p * bins,
                ir_im + p * bins,
                (max_exp - pe) as u32,
            );
        }
    }

    fn fdl_run_ifft_steps(
        &mut self,
        idx: usize,
        step_start: usize,
        step_end: usize,
        commit_output: bool,
    ) {
        if step_start >= step_end {
            return;
        }
        let st = &self.stages[idx];
        let (b, n, bins, mask) = (st.block_size, st.fft_size, st.bins, st.delay_ring_mask);
        let (acc_off, ring_off, ring_len) = (st.acc_off, st.delay_ring_off, st.delay_ring_len);
        let active = st.active_partitions;
        let max_exp = self.acc_max_exp[idx];
        let wp = self.scheduler.stage_state(idx).delay_write_pos;

        if max_exp == i32::MIN || active == 0 {
            if commit_output {
                for k in 0..b {
                    ring_store(self.workspace, ring_off + 2 * ((wp + k) & mask), 0);
                }
                self.scheduler.stage_state_mut(idx).delay_write_pos = wp + b;
            }
            return;
        }

        // Disjoint borrows: acc (in-place inverse buffer) < delay ring.
        let (before_ring, from_ring) = self.workspace.split_at_mut(ring_off);
        let acc = &mut before_ring[acc_off..acc_off + 4 * bins];
        let ring = &mut from_ring[..2 * ring_len];
        let mut e_x = 0;
        if step_start == 0 {
            e_x = acc_i64_to_q31_spectrum_in_place(acc, bins, max_exp).unwrap_or(0);
        }
        let mut sats = 0u64;
        for step in step_start..step_end {
            if let Some(e_t) = q31_irfft_step(n, step, e_x, acc, &mut self.ifft_state[idx]) {
                let sh = e_t + RING_FRAC_BITS as i32;
                for k in 0..b {
                    let (v, sat) =
                        scale_rne_sat(q31_irfft_sample(acc, n, b + k), sh, i64::MIN, i64::MAX);
                    sats += sat as u64;
                    ring_store(ring, 2 * ((wp + k) & mask), v);
                }
                break;
            }
        }
        self.saturation_count = self.saturation_count.saturating_add(sats);
        if commit_output {
            self.scheduler.stage_state_mut(idx).delay_write_pos = wp + b;
        }
    }

    /// Clears every delay line, FDL ring, the shared input history, scheduler cursors and the
    /// saturation counter, keeping the filter.
    pub fn reset(&mut self) {
        if self.shared_in_len > 0 {
            self.workspace[self.shared_in_off..self.shared_in_off + self.shared_in_len].fill(0);
            self.shared_in_write_pos = 0;
        }
        for idx in 0..self.stage_count {
            let st = &mut self.stages[idx];
            if st.kind == 0 {
                self.workspace[st.direct_hist_off..st.direct_hist_off + st.direct_len].fill(0);
                st.direct_head = 0;
            } else {
                let part_bins = st.num_partitions * st.bins;
                self.workspace[st.fdl_re_off..st.fdl_re_off + 2 * part_bins].fill(0);
                self.workspace[st.fdl_exp_off..st.fdl_exp_off + st.num_partitions]
                    .fill(ZERO_SPECTRUM_EXP);
                self.workspace[st.acc_off..st.acc_off + 4 * st.bins].fill(0);
                self.workspace[st.delay_ring_off..st.delay_ring_off + 2 * st.delay_ring_len]
                    .fill(0);
                st.fdl_head = 0;
            }
            self.acc_max_exp[idx] = i32::MIN;
            self.rfft_state[idx] = Q31FftState::new();
            self.ifft_state[idx] = Q31FftState::new();
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

    /// Output samples clipped by saturation since `prepare` or [`Self::reset`].
    #[inline]
    pub const fn saturation_count(&self) -> u64 {
        self.saturation_count
    }

    /// Runtime capabilities metadata.
    #[inline]
    pub const fn capabilities(&self) -> FixedQ31NonUniformCapabilities {
        FixedQ31NonUniformCapabilities {
            stage_count: self.stage_count,
            max_ir_len: self.plan.max_ir_len(),
            active_ir_len: self.active_ir_len,
            latency_samples: 0,
            budget_verdict: self.budget_verdict,
        }
    }
}
