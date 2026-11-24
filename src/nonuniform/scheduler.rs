//! Sample-time deterministic scheduler for Non-Uniform Partitioned Convolution (NUPC).
//!
//! Shared between the `f32` engine (`RTCONV-004B`) and the fixed-point `Q15` engine
//! (`RTCONV-004E`). Operates strictly in **sample time**, never system wall-clock time.
//!
//! ## Time-distributed scheduling & deferral (Wefers 2015 §6.4.3, §6.8)
//! For an FDL stage with block size `B_s`, tap offset `D_s >= B_s`, and causal slack
//! `slack_s = D_s - B_s`:
//! - Effective spreading window: `W_eff = min(slack_s, B_s - 1)` (`span = W_eff + 1` in `1..=B_s`).
//! - Deferral: `defer_s = slack_s.saturating_sub(B_s - 1) = D_s.saturating_sub(2 * B_s - 1)`.
//! - Block `k` (`k = 0, 1, ...`) starts at `t_start(k) = (k + 1) * B_s - 1 + defer_s`
//!   and finishes on or before `t_fin(k) = t_start(k) + W_eff = k * B_s + D_s - 1`,
//!   one sample before `y_s[k * B_s]` is read at `n_due(k) = k * B_s + D_s`.
//! - Total atomic work units `U = rfft_steps + num_partitions + ifft_steps` are distributed
//!   across `slot in 0..span` via the periodic ceiling schedule
//!   `target(slot) = ceil((slot + 1) * U / span)`, which:
//!   1. Always captures the input window on `slot == 0` (`t_start(k)`), keeping the shared
//!      real input ring strictly minimal (`max_s(max(2 * B_s, D_s + 1))`).
//!   2. Keeps consecutive blocks strictly disjoint (`t_fin(k) < t_start(k + 1)`) and shrinks
//!      every stage's output delay ring to power-of-two `2 * B_s`.
//!   3. Eliminates block-boundary load spikes: any contiguous `H`-sample callback (`H = q * B + rem`)
//!      executes at most `q * U + min(U, ceil(rem * U / span))` atomic units regardless of alignment.

use crate::nonuniform::plan::{
    MAX_NONUNIFORM_STAGES, NonUniformPlan, NonUniformStageKind, fdl_stage_op_weights,
};

/// Action emitted by [`NonUniformScheduler`] for an FDL stage at a sample boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageWorkItem {
    /// No FDL work due at this sample.
    None,
    /// Execute a bounded slice of forward RFFT steps (`step_start..step_end`).
    ///
    /// - When `step_start == 0`, the stage captures the completed `2 * B_s` overlap-save window
    ///   from the shared input ring and advances `fdl_head`.
    /// - When `finish_rfft` is `true`, the stage finalizes the spectrum (flushing subnormals in
    ///   `f32` or computing the block product exponent in `Q1.15`) and zeroes the accumulator.
    RfftSteps {
        /// First forward RFFT step index (`0..rfft_steps`).
        step_start: usize,
        /// Exclusive end forward RFFT step index (`step_start < step_end <= rfft_steps`).
        step_end: usize,
        /// Whether this slice finishes the forward RFFT for the current block.
        finish_rfft: bool,
    },
    /// Execute a bounded range of CMAC partitions (`part_start..part_end`) for an in-flight block.
    ContinueCmac {
        /// First partition index to accumulate in this step.
        part_start: usize,
        /// Exclusive end partition index for this step.
        part_end: usize,
        /// If `true`, run the full IFFT and commit output immediately (retained for monolithic
        /// single-step callers; staged schedules emit [`Self::IfftSteps`] or [`Self::CombinedSlice`]).
        finish_ifft: bool,
    },
    /// Execute a bounded slice of inverse IFFT steps (`step_start..step_end`).
    ///
    /// - When `step_start == 0`, the stage prepares the accumulated spectrum for inverse transform.
    /// - When `commit_output` is `true`, the stage commits the `B_s` valid overlap-save output
    ///   samples into the power-of-two `2 * B_s` causal output delay ring.
    IfftSteps {
        /// First inverse IFFT step index (`0..ifft_steps`).
        step_start: usize,
        /// Exclusive end inverse IFFT step index (`step_start < step_end <= ifft_steps`).
        step_end: usize,
        /// Whether this slice completes the inverse IFFT and commits to the output delay ring.
        commit_output: bool,
    },
    /// Multi-phase slice executed on a single sample when `total_units > span` (for example
    /// `slack == 0`, very small slack relative to partition count, or a non-staged FFT backend).
    ///
    /// Phases execute strictly in causal order: RFFT (`rfft_start..rfft_end`), then CMAC
    /// (`cmac_start..cmac_end`), then IFFT (`ifft_start..ifft_end`, committing output when
    /// `commit_output` is `true`).
    CombinedSlice {
        /// First forward RFFT step (`rfft_start == rfft_end` if no RFFT work).
        rfft_start: usize,
        /// Exclusive end forward RFFT step.
        rfft_end: usize,
        /// Whether `rfft_end` completes the forward RFFT for the current block.
        finish_rfft: bool,
        /// First CMAC partition (`cmac_start == cmac_end` if no CMAC work).
        cmac_start: usize,
        /// Exclusive end CMAC partition.
        cmac_end: usize,
        /// First inverse IFFT step (`ifft_start == ifft_end` if no IFFT work).
        ifft_start: usize,
        /// Exclusive end inverse IFFT step.
        ifft_end: usize,
        /// Whether `ifft_end` completes the inverse IFFT and commits output to the delay ring.
        commit_output: bool,
    },
}

impl StageWorkItem {
    /// Returns `true` if this item starts a new block's forward RFFT (`step 0`).
    #[inline]
    pub const fn starts_block(self) -> bool {
        match self {
            Self::RfftSteps { step_start, .. } => step_start == 0,
            Self::CombinedSlice {
                rfft_start,
                rfft_end,
                ..
            } => rfft_start == 0 && rfft_end > 0,
            _ => false,
        }
    }

    /// Returns `true` if this item commits the completed block into the output delay ring.
    #[inline]
    pub const fn commits_output(self) -> bool {
        match self {
            Self::ContinueCmac { finish_ifft, .. } => finish_ifft,
            Self::IfftSteps { commit_output, .. } | Self::CombinedSlice { commit_output, .. } => {
                commit_output
            }
            Self::None | Self::RfftSteps { .. } => false,
        }
    }
}

/// Per-stage sample-time scheduling state (zero heap).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageScheduleState {
    /// Number of new input samples buffered toward the next `block_size` boundary (`0..block_size`).
    pub input_fill: usize,
    /// Total samples written into this stage's output delay ring.
    pub delay_write_pos: usize,
    /// Next sample index to read from this stage's output delay ring.
    pub delay_read_pos: usize,
    /// Samples remaining until the next deferred block trigger `t_start(k)`.
    pub next_start_countdown: usize,
    /// Whether an in-flight FDL block is currently active across RFFT/CMAC/IFFT phases.
    pub in_flight: bool,
    /// Next CMAC partition index to accumulate for the in-flight block.
    pub next_part: usize,
    /// Atomic work units completed for the in-flight block (`0..=total_units`).
    pub units_done: usize,
    /// Total atomic work units scheduled for the in-flight block (`R + P + I`).
    pub total_units: usize,
    /// Samples elapsed since the in-flight block's deferred start `t_start(k)` (`0..=W_eff`).
    pub samples_since_ready: usize,
    /// Number of active work calls executed for the in-flight block.
    pub steps_done: usize,
    /// Number of deadline misses observed (must remain `0` for valid causal plans).
    pub deadline_misses: u64,
}

/// Computes the deterministic multiply-accumulate-equivalent operation count of a single
/// [`StageWorkItem`] emitted for FDL stage `stage_idx` of `plan`, using the exact same
/// operation weights as [`NonUniformPlan::estimated_worst_ops_per_callback`].
pub fn stage_work_item_ops(
    plan: &NonUniformPlan,
    stage_idx: usize,
    item: StageWorkItem,
    rfft_steps: usize,
    ifft_steps: usize,
) -> usize {
    if matches!(item, StageWorkItem::None) {
        return 0;
    }
    let st = &plan.stages()[stage_idx];
    if st.kind != NonUniformStageKind::Fdl {
        return 0;
    }
    let Ok((rfft_step_ops, ifft_step_ops, partition_ops)) =
        fdl_stage_op_weights(st.block_size, rfft_steps, ifft_steps)
    else {
        return 0;
    };
    match item {
        StageWorkItem::None => 0,
        StageWorkItem::RfftSteps {
            step_start,
            step_end,
            ..
        } => step_end.saturating_sub(step_start) * rfft_step_ops,
        StageWorkItem::ContinueCmac {
            part_start,
            part_end,
            finish_ifft,
        } => {
            let cmac = part_end.saturating_sub(part_start) * partition_ops;
            if finish_ifft {
                cmac + ifft_steps.max(1) * ifft_step_ops
            } else {
                cmac
            }
        }
        StageWorkItem::IfftSteps {
            step_start,
            step_end,
            ..
        } => step_end.saturating_sub(step_start) * ifft_step_ops,
        StageWorkItem::CombinedSlice {
            rfft_start,
            rfft_end,
            cmac_start,
            cmac_end,
            ifft_start,
            ifft_end,
            ..
        } => {
            rfft_end.saturating_sub(rfft_start) * rfft_step_ops
                + cmac_end.saturating_sub(cmac_start) * partition_ops
                + ifft_end.saturating_sub(ifft_start) * ifft_step_ops
        }
    }
}

/// Representation-independent sample-time scheduler shared across `f32` and `Q15` NUPC engines.
#[derive(Clone, Copy, Debug)]
pub struct NonUniformScheduler {
    states: [StageScheduleState; MAX_NONUNIFORM_STAGES],
    stage_count: usize,
    sample_cursor: u64,
}

impl NonUniformScheduler {
    /// Initializes a clean scheduler for a validated [`NonUniformPlan`].
    pub fn new(plan: &NonUniformPlan) -> Self {
        let mut sched = Self {
            states: [StageScheduleState::default(); MAX_NONUNIFORM_STAGES],
            stage_count: plan.stage_count(),
            sample_cursor: 0,
        };
        sched.reset(plan);
        sched
    }

    /// Resets all stage cursors and pre-seeds each FDL stage's output delay ring with
    /// `ir_offset` initial virtual zero samples so output sample `y_s[0]` aligns exactly
    /// with global sample `t = D_s`, while deferring block `0` to `t_start(0) = D_s - 1 - W_eff`.
    pub fn reset(&mut self, plan: &NonUniformPlan) {
        self.sample_cursor = 0;
        for (idx, st) in plan.stages().iter().enumerate() {
            let mut s = StageScheduleState::default();
            if st.kind == NonUniformStageKind::Fdl {
                let w_eff = st.slack_samples.min(st.block_size.saturating_sub(1));
                // Block 0 triggers on call index `t_start(0) = D_s - 1 - W_eff = B_s - 1 + defer_s`
                // (i.e. after `D_s - W_eff` sample arrivals). Subsequent blocks trigger every `B_s`.
                s.next_start_countdown = st.ir_offset - w_eff;
                s.delay_write_pos = st.ir_offset;
                s.delay_read_pos = 0;
            }
            self.states[idx] = s;
        }
    }

    /// Advances the global sample counter by 1.
    #[inline]
    pub fn advance_sample(&mut self) {
        self.sample_cursor = self.sample_cursor.wrapping_add(1);
    }

    /// Current global sample index.
    #[inline]
    pub const fn sample_cursor(&self) -> u64 {
        self.sample_cursor
    }

    /// Total deadline misses recorded across all stages (always `0` on valid causal plans).
    pub fn total_deadline_misses(&self) -> u64 {
        let mut total = 0u64;
        for s in &self.states[..self.stage_count] {
            total = total.saturating_add(s.deadline_misses);
        }
        total
    }

    /// Immutable access to a stage's state.
    #[inline]
    pub fn stage_state(&self, idx: usize) -> &StageScheduleState {
        &self.states[idx]
    }

    /// Mutable access to a stage's state.
    #[inline]
    pub fn stage_state_mut(&mut self, idx: usize) -> &mut StageScheduleState {
        &mut self.states[idx]
    }

    /// Records one newly arrived input sample for FDL stage `idx` using the default `f32`
    /// staged transform step count (`log2(block_size)` forward and inverse steps) and returns
    /// any FDL work due at this exact sample boundary.
    #[inline]
    pub fn on_fdl_sample(&mut self, idx: usize, plan: &NonUniformPlan) -> StageWorkItem {
        let steps = plan.stages()[idx].block_size.trailing_zeros() as usize;
        self.on_fdl_sample_with_steps(idx, plan, steps, steps)
    }

    /// Records one newly arrived input sample for FDL stage `idx` with explicit transform
    /// step counts (`rfft_steps`, `ifft_steps`, each `>= 1`) and returns any FDL work due at
    /// this exact sample boundary.
    pub fn on_fdl_sample_with_steps(
        &mut self,
        idx: usize,
        plan: &NonUniformPlan,
        rfft_steps: usize,
        ifft_steps: usize,
    ) -> StageWorkItem {
        let st = &plan.stages()[idx];
        let s = &mut self.states[idx];

        let b = st.block_size;
        let p = st.num_partitions;
        let r = rfft_steps.max(1);
        let i = ifft_steps.max(1);
        let total_units = r + p + i;
        let w_eff = st.slack_samples.min(b.saturating_sub(1));
        let span = w_eff + 1;

        s.input_fill += 1;
        if s.input_fill >= b {
            s.input_fill = 0;
        }

        if s.next_start_countdown > 0 {
            s.next_start_countdown -= 1;
        }

        if s.next_start_countdown == 0 {
            s.next_start_countdown = b;
            if s.in_flight {
                // Should never happen for valid causal plans; record miss explicitly.
                s.deadline_misses = s.deadline_misses.saturating_add(1);
            }
            s.in_flight = true;
            s.units_done = 0;
            s.total_units = total_units;
            s.samples_since_ready = 0;
            s.steps_done = 0;
            s.next_part = 0;
        } else if s.in_flight {
            s.samples_since_ready += 1;
        } else {
            return StageWorkItem::None;
        }

        let slot = s.samples_since_ready;
        let u0 = s.units_done;
        let target = if slot >= w_eff || span <= 1 {
            total_units
        } else {
            ((slot + 1) * total_units).div_ceil(span).min(total_units)
        };

        if target <= u0 {
            return StageWorkItem::None;
        }

        s.units_done = target;
        s.steps_done += 1;

        let rfft_start = u0.min(r);
        let rfft_end = target.min(r);
        let cmac_start = u0.saturating_sub(r).min(p);
        let cmac_end = target.saturating_sub(r).min(p);
        let ifft_start = u0.saturating_sub(r + p).min(i);
        let ifft_end = target.saturating_sub(r + p).min(i);

        s.next_part = cmac_end.max(s.next_part);
        let finish_rfft = rfft_start < rfft_end && rfft_end == r;
        let commit_output = ifft_start < ifft_end && ifft_end == i;
        if target >= total_units {
            s.in_flight = false;
        }

        let has_rfft = rfft_start < rfft_end;
        let has_cmac = cmac_start < cmac_end;
        let has_ifft = ifft_start < ifft_end;

        match (has_rfft, has_cmac, has_ifft) {
            (true, false, false) => StageWorkItem::RfftSteps {
                step_start: rfft_start,
                step_end: rfft_end,
                finish_rfft,
            },
            (false, true, false) => StageWorkItem::ContinueCmac {
                part_start: cmac_start,
                part_end: cmac_end,
                finish_ifft: false,
            },
            (false, false, true) => StageWorkItem::IfftSteps {
                step_start: ifft_start,
                step_end: ifft_end,
                commit_output,
            },
            _ => StageWorkItem::CombinedSlice {
                rfft_start,
                rfft_end,
                finish_rfft,
                cmac_start,
                cmac_end,
                ifft_start,
                ifft_end,
                commit_output,
            },
        }
    }
}
