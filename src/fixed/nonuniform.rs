//! Integrated Fixed-Point Q1.15/BFP Non-Uniform Partitioned Convolver (`FixedQ15NonUniformConvolver`).
//!
//! Shares the exact representation-independent `Planning Plane` ([`NonUniformPlan`],
//! [`NonUniformBudget`]) and `Streaming Plane` ([`NonUniformScheduler`]) from `crate::nonuniform`,
//! paired with the Q1.15/BFP `Numeric Plane` from `crate::fixed`.

use crate::fixed::block_float::{Q31Taps, TapSource};
use crate::fixed::fft_q15::fixed_fft_supports_size;
use crate::fixed::fft_q31::{
    Q31FftState, ZERO_SPECTRUM_EXP, acc_i64_to_q31_spectrum_in_place, q31_fft_steps,
    q31_irfft_sample, q31_irfft_step, q31_rfft_step, q31_spectrum_to_q15, scale_rne_sat,
};
use crate::fixed::q15::round_even_sat_q15_from_i64;
use crate::fixed::spectral::{cmac_guard_bits, cmac_partition, exp_to_word, prepare_partition};
use crate::nonuniform::memory::{shared_input_ring_len, stage_delay_ring_len};
use crate::nonuniform::{
    BudgetVerdict, MAX_NONUNIFORM_STAGES, NonUniformBudget, NonUniformPlan, NonUniformScheduler,
    NonUniformStageKind, StageWorkItem,
};
use crate::{ConvError, FftSize};

/// Extra fractional bits carried by FDL stage outputs in the causal delay rings (`Q1.23` in
/// `i32`, stored as two `i16` words). All stage contributions and the direct head are summed
/// at this precision and rounded to Q1.15 exactly once per output sample (convergent RNE),
/// instead of rounding every stage separately (which costs ~`10*log10(stages)` dB of SNR).
const RING_FRAC_BITS: u32 = 8;

#[inline(always)]
fn ring_load(ws: &[i16], off: usize) -> i32 {
    ((ws[off] as i32) << 16) | (ws[off + 1] as u16 as i32)
}

#[inline(always)]
fn ring_store(ws: &mut [i16], off: usize, v: i32) {
    ws[off] = (v >> 16) as i16;
    ws[off + 1] = v as i16;
}

/// Runs forward-transform `steps` of one FDL stage in place inside its accumulator region
/// (`acc_off`, `8 * bins` words) from the shared overlap-save window at `time_off`, and on the
/// last step stores the Q1.15 BFP spectrum at `re_off` / `im_off`, returning its exponent.
/// Inlined into the caller, which keeps only the offsets it needs live (no stage-layout copy),
/// so the per-sample call chain stays shallow on RV32.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn rfft_steps_into_slot(
    ws: &mut [i16],
    state: &mut Q31FftState,
    n: usize,
    steps: core::ops::Range<usize>,
    time_off: usize,
    acc_off: usize,
    re_off: usize,
    im_off: usize,
) -> Option<i32> {
    let bins = n / 2 + 1;
    let mut finished = None;
    for step in steps {
        // Disjoint borrows: ... < acc (in-place buffer) < time window; slots precede acc.
        let (head, from_acc) = ws.split_at_mut(acc_off);
        let (acc, from_time) = from_acc.split_at_mut(time_off - acc_off);
        let time_in = &from_time[..n];
        if let Some(e) = q31_rfft_step(
            n,
            step,
            |i| time_in[i] as i32,
            -15,
            &mut acc[..8 * bins],
            state,
        ) {
            let (lo, hi) = head.split_at_mut(im_off);
            finished = Some(q31_spectrum_to_q15(
                &acc[..8 * bins],
                n,
                e,
                &mut lo[re_off..re_off + bins],
                &mut hi[..bins],
            ));
        }
    }
    finished
}

/// Explicit memory breakdown for [`FixedQ15NonUniformConvolver`] in `i16` elements and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ15NonUniformMemoryRequirements {
    /// Time-domain state buffers (`i16`: direct FIR history + coefficients, plus the single
    /// shared Q1.15 real input history ring for all FDL stages).
    pub state_i16: usize,
    /// Frequency-domain BFP spectra, exponents, and per-stage `i64` accumulators (`i16`;
    /// each stage's `8 * (B_s + 1) = 4 * N_s + 8` accumulator words also host the in-place
    /// staged 32-bit-mantissa FFT (`2 * N_s + 4` words) and IFFT (`4 * N_s + 4` words)).
    pub spectral_i16: usize,
    /// Per-stage power-of-two causal output delay rings (`4 * B_s` `i16` words per FDL stage;
    /// `2 * B_s` samples in `Q1.23` stored as two `i16` words each).
    pub delay_i16: usize,
    /// Shared overlap-save time window of the largest FDL stage (`max N_s` `i16` words).
    pub scratch_i16: usize,
    /// Total required `&mut [i16]` elements.
    pub total_i16: usize,
    /// Total required workspace size in bytes (`total_i16 * 2`).
    pub total_bytes: usize,
    /// Size in bytes of the [`FixedQ15NonUniformConvolver`] value itself.
    /// Per-stage `i64` spectral accumulators and FFT scratch live in the caller workspace, so
    /// the instance holds only stage metadata and scheduler state; it is still included in
    /// budget admission checks.
    pub instance_bytes: usize,
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
    ir_re_off: usize,
    ir_im_off: usize,
    fdl_re_off: usize,
    fdl_im_off: usize,
    ir_exp_off: usize,
    fdl_exp_off: usize,
    acc_re_off: usize,
    acc_im_off: usize,
    delay_ring_off: usize,
    delay_ring_len: usize,
    delay_ring_mask: usize,
    defer_samples: usize,
    fft_steps: usize,
    fdl_head: usize,
    block_size: usize,
    fft_size: usize,
    bins: usize,
    num_partitions: usize,
    active_partitions: usize,
}

/// Zero-latency, caller-allocated Fixed-Point Q1.15/BFP Non-Uniform Partitioned Convolver.
///
/// Supports FDL stage FFT sizes `N64` through `N2048` (`block_size` `32..=1024`) with
/// time-distributed BFP FFT/CMAC/IFFT execution and power-of-two `2 * B_s` Q1.23 output rings.
/// Plans with larger FDL FFT stages return [`ConvError::UnsupportedFftSize`].
pub struct FixedQ15NonUniformConvolver<'a> {
    plan: NonUniformPlan,
    shared_in_off: usize,
    shared_in_len: usize,
    shared_in_write_pos: usize,
    stages: [FixedStageLayout; MAX_NONUNIFORM_STAGES],
    stage_count: usize,
    scheduler: NonUniformScheduler,
    workspace: &'a mut [i16],
    acc_max_exp: [i32; MAX_NONUNIFORM_STAGES],
    rfft_state: [Q31FftState; MAX_NONUNIFORM_STAGES],
    ifft_state: [Q31FftState; MAX_NONUNIFORM_STAGES],
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
        let mut state_i16 = shared_input_ring_len(plan)?;
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
                    if !fixed_fft_supports_size(fft) {
                        return Err(ConvError::UnsupportedFftSize {
                            fft_size: st.fft_size,
                        });
                    }
                    let bins = fft.spectrum_bins();
                    let part_bins = st
                        .num_partitions
                        .checked_mul(bins)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    // ir_re, ir_im, fdl_re, fdl_im + ir_exp(P) + fdl_exp(P)
                    // + acc_re(4*bins) + acc_im(4*bins) (also holds 4*N in-place staged FFT/IFFT)
                    let stage_spec = part_bins
                        .checked_mul(4)
                        .and_then(|v| v.checked_add(st.num_partitions.checked_mul(2)?))
                        .and_then(|v| v.checked_add(bins.checked_mul(8)?))
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    spectral_i16 = spectral_i16
                        .checked_add(stage_spec)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let dlen = stage_delay_ring_len(st.block_size)?;
                    delay_i16 = delay_i16
                        .checked_add(dlen.checked_mul(2).ok_or(ConvError::ArithmeticOverflow)?)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    max_fft = max_fft.max(st.fft_size);
                    max_bins = max_bins.max(bins);
                }
            }
        }

        // Shared scratch: the overlap-save time window (max_fft). Per-stage transforms run in
        // place inside acc_re + acc_im (8*bins = 4*N + 8 words): the forward FFT uses 2*N + 4,
        // the inverse normalizes the i64 accumulators in place and uses 4*N + 4.
        let _ = max_bins;
        let scratch_i16 = max_fft;
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
            instance_bytes: core::mem::size_of::<FixedQ15NonUniformConvolver<'static>>(),
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
        Self::prepare_from(&ir, plan, budget, workspace)
    }

    /// Prepares from Q1.31 taps (`tap / 2^31`, e.g. 24/32-bit PCM or float FIR files).
    ///
    /// The direct head is rounded to Q1.15; every FDL partition is normalized to the full Q1.15
    /// mantissa before its FFT and keeps its own block exponent, so low-level IR tails retain
    /// ~15 significant bits instead of a few LSBs of the absolute Q1.15 grid.
    pub fn prepare_q31(
        ir: &[i32],
        plan: NonUniformPlan,
        budget: Option<NonUniformBudget>,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        Self::prepare_q31_with(ir.len(), |i| ir[i], plan, budget, workspace)
    }

    /// Like [`Self::prepare_q31`] with taps supplied by index (`tap(i)` for `i < len`), so they
    /// can be decoded from any storage without an intermediate `&[i32]`.
    pub fn prepare_q31_with<F: Fn(usize) -> i32>(
        len: usize,
        tap: F,
        plan: NonUniformPlan,
        budget: Option<NonUniformBudget>,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        Self::prepare_from(&Q31Taps { len, tap }, plan, budget, workspace)
    }

    fn prepare_from<S: TapSource>(
        ir: &S,
        plan: NonUniformPlan,
        budget: Option<NonUniformBudget>,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        if ir.len() == 0 {
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
                let v = plan.check_budget(b, req.total_bytes.saturating_add(req.instance_bytes));
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

        let shared_in_off = 0usize;
        let shared_in_len = shared_input_ring_len(&plan)?;
        let mut stages = [FixedStageLayout::default(); MAX_NONUNIFORM_STAGES];
        let mut cursor = shared_in_len;
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
                    let dlen = stage_delay_ring_len(st.block_size)?;
                    let defer_samples = st
                        .slack_samples
                        .saturating_sub(st.block_size.saturating_sub(1));
                    let fft_steps = q31_fft_steps(fft.len());

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
                    let acc_re_off = cursor;
                    cursor += 4 * bins;
                    let acc_im_off = cursor;
                    cursor += 4 * bins;
                    let delay_ring_off = cursor;
                    cursor += 2 * dlen;

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
                        ir_re_off,
                        ir_im_off,
                        fdl_re_off,
                        fdl_im_off,
                        ir_exp_off,
                        fdl_exp_off,
                        acc_re_off,
                        acc_im_off,
                        delay_ring_off,
                        delay_ring_len: dlen,
                        delay_ring_mask: dlen - 1,
                        defer_samples,
                        fft_steps,
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

        let _ = max_bins;
        let time_scratch_off = cursor;
        cursor += max_fft;
        workspace[..cursor].fill(0);

        // Pre-load direct coefficients and pre-transform all FDL IR partitions into Q15 BFP,
        // reusing each stage's contiguous `acc_re + acc_im` (`8 * bins >= 4 * N`) as FFT scratch.
        for (idx, st_plan) in plan.stages().iter().enumerate() {
            let st = &stages[idx];
            match st_plan.kind {
                NonUniformStageKind::Direct => {
                    for k in 0..st.direct_active_len {
                        workspace[st.direct_coeffs_off + k] = ir.q15(st_plan.ir_offset + k);
                    }
                }
                NonUniformStageKind::Fdl => {
                    let b = st.block_size;
                    let n = st.fft_size;
                    let bins = st.bins;
                    for p in 0..st.active_partitions {
                        let start = st_plan.ir_offset + p * b;
                        let end = (start + b)
                            .min(st_plan.ir_offset + st_plan.ir_len)
                            .min(ir.len());
                        let len = end.saturating_sub(start);
                        // Disjoint borrows: ir_re < ir_im < acc (forward FFT buffer).
                        let (before_im, from_im) = workspace.split_at_mut(st.ir_im_off);
                        let (im_region, from_acc) =
                            from_im.split_at_mut(st.acc_re_off - st.ir_im_off);
                        let re_slot =
                            &mut before_im[st.ir_re_off + p * bins..st.ir_re_off + (p + 1) * bins];
                        let im_slot = &mut im_region[p * bins..(p + 1) * bins];
                        let e = prepare_partition(
                            ir,
                            start,
                            len,
                            n,
                            &mut from_acc[..8 * bins],
                            re_slot,
                            im_slot,
                        );
                        workspace[st.ir_exp_off + p] = exp_to_word(e);
                    }
                    workspace[st.acc_re_off..st.acc_re_off + 8 * bins].fill(0);
                    // Empty FDL slots never set the CMAC alignment exponent.
                    workspace[st.fdl_exp_off..st.fdl_exp_off + st.num_partitions]
                        .fill(exp_to_word(ZERO_SPECTRUM_EXP));
                }
            }
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
        let in_write_pos = self.shared_in_write_pos;
        if self.shared_in_len > 0 {
            self.workspace[self.shared_in_off + in_write_pos] = x;
        }

        // Accumulate every stage at Q2.30 and round to Q1.15 exactly once (convergent RNE).
        let mut sum: i64 = 0;

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
                    sum += acc;
                }
            } else {
                let read_pos = self.scheduler.stage_state(idx).delay_read_pos;
                let ring_idx = read_pos & self.stages[idx].delay_ring_mask;
                let off = self.stages[idx].delay_ring_off + 2 * ring_idx;
                let y_stage = ring_load(self.workspace, off);
                ring_store(self.workspace, off, 0);
                self.scheduler.stage_state_mut(idx).delay_read_pos = read_pos + 1;
                sum += (y_stage as i64) << (15 - RING_FRAC_BITS);

                let steps = self.stages[idx].fft_steps;
                let work = self
                    .scheduler
                    .on_fdl_sample_with_steps(idx, &self.plan, steps, steps);
                match work {
                    StageWorkItem::None => {}
                    StageWorkItem::RfftSteps {
                        step_start,
                        step_end,
                        finish_rfft,
                    } => {
                        self.fdl_run_rfft_steps(
                            idx,
                            in_write_pos,
                            step_start,
                            step_end,
                            finish_rfft,
                        )?;
                    }
                    StageWorkItem::ContinueCmac {
                        part_start,
                        part_end,
                        finish_ifft,
                    } => {
                        self.fdl_run_cmac(idx, part_start, part_end);
                        if finish_ifft {
                            self.fdl_run_ifft_steps(idx, 0, steps, true)?;
                        }
                    }
                    StageWorkItem::IfftSteps {
                        step_start,
                        step_end,
                        commit_output,
                    } => {
                        self.fdl_run_ifft_steps(idx, step_start, step_end, commit_output)?;
                    }
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
                            )?;
                        }
                        if cmac_start < cmac_end {
                            self.fdl_run_cmac(idx, cmac_start, cmac_end);
                        }
                        if ifft_start < ifft_end {
                            self.fdl_run_ifft_steps(idx, ifft_start, ifft_end, commit_output)?;
                        }
                    }
                }
            }
        }

        if self.shared_in_len > 0 {
            let next = in_write_pos + 1;
            self.shared_in_write_pos = if next == self.shared_in_len { 0 } else { next };
        }
        self.scheduler.advance_sample();
        let (y, sat) = round_even_sat_q15_from_i64(sum);
        if sat {
            self.saturation_count = self.saturation_count.saturating_add(1);
        }
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
    ) -> Result<(), ConvError> {
        let active_p = self.stages[idx].active_partitions;
        if active_p == 0 || step_start >= step_end {
            return Ok(());
        }

        let n = self.stages[idx].fft_size;
        let bins = self.stages[idx].bins;
        let p_total = self.stages[idx].num_partitions;
        let defer_samples = self.stages[idx].defer_samples;

        if step_start == 0 {
            let next_head = if self.stages[idx].fdl_head + 1 == p_total {
                0
            } else {
                self.stages[idx].fdl_head + 1
            };
            self.stages[idx].fdl_head = next_head;
            self.copy_shared_input_window(in_write_pos, defer_samples, n);
        }

        // Only the offsets used here are read (copying the whole stage layout costs ~100 B of
        // stack on RV32).
        let slot = self.stages[idx].fdl_head;
        let fdl_re_off = self.stages[idx].fdl_re_off;
        let fdl_exp_off = self.stages[idx].fdl_exp_off;
        let acc_off = self.stages[idx].acc_re_off;

        let finished_exp = rfft_steps_into_slot(
            self.workspace,
            &mut self.rfft_state[idx],
            n,
            step_start..step_end,
            self.time_scratch_off,
            acc_off,
            fdl_re_off + slot * bins,
            self.stages[idx].fdl_im_off + slot * bins,
        );

        if finish_rfft {
            if let Some(x_exp) = finished_exp {
                self.workspace[fdl_exp_off + slot] = exp_to_word(x_exp);
            }
            // Pre-compute max product exponent across all active partitions for this block so
            // distributed CMAC steps align identically to burst mode.
            let ir_exp_off = self.stages[idx].ir_exp_off;
            let mut max_exp = i32::MIN;
            for p in 0..active_p {
                let s = if slot >= p {
                    slot - p
                } else {
                    slot + p_total - p
                };
                let pe = (self.workspace[fdl_exp_off + s] as i32)
                    + (self.workspace[ir_exp_off + p] as i32);
                if pe > max_exp {
                    max_exp = pe;
                }
            }
            self.acc_max_exp[idx] = max_exp;
            self.workspace[acc_off..acc_off + 8 * bins].fill(0);
        }

        Ok(())
    }

    fn fdl_run_cmac(&mut self, idx: usize, part_start: usize, part_end: usize) {
        let st = self.stages[idx];
        let bins = st.bins;
        let p_total = st.num_partitions;
        let max_exp = self.acc_max_exp[idx];
        let p_end = part_end.min(st.active_partitions);

        if max_exp == i32::MIN || part_start >= p_end {
            return;
        }

        let guard = cmac_guard_bits(st.active_partitions) as i32;
        for p in part_start..p_end {
            let s = if st.fdl_head >= p {
                st.fdl_head - p
            } else {
                st.fdl_head + p_total - p
            };
            let pe = (self.workspace[st.fdl_exp_off + s] as i32)
                + (self.workspace[st.ir_exp_off + p] as i32);
            cmac_partition(
                self.workspace,
                st.acc_re_off,
                bins,
                st.fdl_re_off + s * bins,
                st.fdl_im_off + s * bins,
                st.ir_re_off + p * bins,
                st.ir_im_off + p * bins,
                guard - (max_exp - pe),
            );
        }
    }

    fn fdl_run_ifft_steps(
        &mut self,
        idx: usize,
        step_start: usize,
        step_end: usize,
        commit_output: bool,
    ) -> Result<(), ConvError> {
        if step_start >= step_end {
            return Ok(());
        }

        let st = self.stages[idx];
        let b = st.block_size;
        let n = st.fft_size;
        let bins = st.bins;
        let mask = st.delay_ring_mask;
        let max_exp = self.acc_max_exp[idx];
        let wp = self.scheduler.stage_state(idx).delay_write_pos;

        if max_exp == i32::MIN || st.active_partitions == 0 {
            if commit_output {
                for k in 0..b {
                    let r = (wp + k) & mask;
                    ring_store(self.workspace, st.delay_ring_off + 2 * r, 0);
                }
                self.scheduler.stage_state_mut(idx).delay_write_pos = wp + b;
            }
            return Ok(());
        }

        // Disjoint borrows: acc (in-place inverse buffer) < delay ring.
        let (before_ring, from_ring) = self.workspace.split_at_mut(st.delay_ring_off);
        let acc = &mut before_ring[st.acc_re_off..st.acc_re_off + 8 * bins];
        let delay_ring = &mut from_ring[..2 * st.delay_ring_len];
        let mut e_x = 0;
        if step_start == 0 {
            // Normalize the exact i64 accumulators in place to i32 mantissas (all-zero
            // accumulators are already a zero spectrum in the inverse layout).
            let guard = cmac_guard_bits(st.active_partitions) as i32;
            e_x = acc_i64_to_q31_spectrum_in_place(acc, bins, max_exp - 30 - guard).unwrap_or(0);
        }

        let mut total_sats = 0u64;
        for step in step_start..step_end {
            if let Some(e_t) = q31_irfft_step(n, step, e_x, acc, &mut self.ifft_state[idx]) {
                // Valid overlap-save half x[B..2B] into the Q1.23 ring, one RNE rounding.
                let sh = e_t + 15 + RING_FRAC_BITS as i32;
                for k in 0..b {
                    let (v, sat) = scale_rne_sat(
                        q31_irfft_sample(acc, n, b + k),
                        sh,
                        i32::MIN as i64,
                        i32::MAX as i64,
                    );
                    total_sats += sat as u64;
                    let r = (wp + k) & mask;
                    ring_store(delay_ring, 2 * r, v as i32);
                }
                break;
            }
        }
        self.saturation_count = self.saturation_count.saturating_add(total_sats);

        if commit_output {
            self.scheduler.stage_state_mut(idx).delay_write_pos = wp + b;
        }

        Ok(())
    }

    /// Deterministically resets all delay lines, FDL rings, shared input history, scheduler
    /// cursors, and saturation counter.
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
                self.workspace[st.fdl_re_off..st.fdl_re_off + part_bins].fill(0);
                self.workspace[st.fdl_im_off..st.fdl_im_off + part_bins].fill(0);
                self.workspace[st.fdl_exp_off..st.fdl_exp_off + st.num_partitions]
                    .fill(exp_to_word(ZERO_SPECTRUM_EXP));
                self.workspace[st.acc_re_off..st.acc_re_off + 4 * st.bins].fill(0);
                self.workspace[st.acc_im_off..st.acc_im_off + 4 * st.bins].fill(0);
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
