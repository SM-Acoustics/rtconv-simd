//! Sample-time deterministic scheduler for Non-Uniform Partitioned Convolution (NUPC).
//!
//! Shared between the `f32` engine (`RTCONV-004B`) and the fixed-point `Q15` engine
//! (`RTCONV-004E`). Operates strictly in **sample time**, never system wall-clock time.

use crate::nonuniform::plan::{MAX_NONUNIFORM_STAGES, NonUniformPlan, NonUniformStageKind};

/// Action emitted by [`NonUniformScheduler`] for an FDL stage at a sample boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageWorkItem {
    /// No FDL work due at this sample.
    None,
    /// Execute RFFT of the newly completed `block_size` input frame, push into FDL ring,
    /// and begin CMAC accumulation (`partition_start..partition_end`).
    StartBlockAndCmac {
        /// First partition index to accumulate in this step.
        part_start: usize,
        /// Exclusive end partition index for this step.
        part_end: usize,
        /// If true, run IFFT immediately and commit the `block_size` output into the delay ring.
        finish_ifft: bool,
    },
    /// Continue spread CMAC accumulation (`part_start..part_end`) for an in-flight block.
    ContinueCmac {
        /// First partition index to accumulate in this step.
        part_start: usize,
        /// Exclusive end partition index for this step.
        part_end: usize,
        /// If true, run IFFT and commit the `block_size` output into the delay ring.
        finish_ifft: bool,
    },
}

/// Per-stage sample-time scheduling state (zero heap).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageScheduleState {
    /// Number of new input samples buffered toward the next `block_size` trigger (`0..block_size`).
    pub input_fill: usize,
    /// Total samples written into this stage's output delay ring.
    pub delay_write_pos: usize,
    /// Next sample index to read from this stage's output delay ring.
    pub delay_read_pos: usize,
    /// Whether an in-flight spread FDL block is currently accumulating CMAC partitions.
    pub in_flight: bool,
    /// Next partition index to accumulate for the in-flight block.
    pub next_part: usize,
    /// Samples elapsed since the in-flight block's `t_ready` trigger.
    pub samples_since_ready: usize,
    /// Number of spread sub-steps completed for the in-flight block.
    pub steps_done: usize,
    /// Number of deadline misses observed (must remain `0` for valid causal plans).
    pub deadline_misses: u64,
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
    /// `ir_offset` initial zero samples so output sample `y_s[0]` aligns exactly with global
    /// sample `t = D_s`.
    pub fn reset(&mut self, plan: &NonUniformPlan) {
        self.sample_cursor = 0;
        for (idx, st) in plan.stages().iter().enumerate() {
            let mut s = StageScheduleState::default();
            if st.kind == NonUniformStageKind::Fdl {
                // Pre-fill `D_s` zero samples ahead of the first computed FDL output block.
                // At t = 0..D_s-1, reading from the delay ring yields 0.0.
                // At t = B_s, block 0 is ready and written starting at ring position D_s.
                // Since D_s >= B_s, the write completes on or before t = D_s!
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

    /// Records one newly arrived input sample for FDL stage `idx` and returns any FDL work
    /// due at this exact sample boundary.
    pub fn on_fdl_sample(&mut self, idx: usize, plan: &NonUniformPlan) -> StageWorkItem {
        let st = &plan.stages()[idx];
        let s = &mut self.states[idx];

        s.input_fill += 1;

        // If a previous spread block is still in flight, check whether the next spread step
        // or its hard deadline (`effective_window = min(slack_samples, block_size - 1)`) has arrived.
        if s.in_flight {
            s.samples_since_ready += 1;
            let effective_window = st.slack_samples.min(st.block_size.saturating_sub(1));
            let steps = st.spread_steps.max(1);
            let step_interval = if steps > 1 && effective_window > 0 {
                (effective_window / (steps - 1)).max(1)
            } else {
                1
            };
            let deadline_reached = s.samples_since_ready >= effective_window;
            let next_step_due = s.steps_done < steps
                && s.samples_since_ready >= s.steps_done.saturating_mul(step_interval);

            if deadline_reached || next_step_due {
                let start = s.next_part;
                let parts_per_step = st.num_partitions.div_ceil(steps);
                let end = if deadline_reached || s.steps_done + 1 >= steps {
                    st.num_partitions
                } else {
                    (start + parts_per_step).min(st.num_partitions)
                };
                s.next_part = end;
                s.steps_done += 1;
                let finish_ifft = end >= st.num_partitions;
                if finish_ifft {
                    s.in_flight = false;
                }
                return StageWorkItem::ContinueCmac {
                    part_start: start,
                    part_end: end,
                    finish_ifft,
                };
            }
        }

        if s.input_fill >= st.block_size {
            s.input_fill = 0;
            if s.in_flight {
                // Should never happen for valid causal plans; record miss explicitly.
                s.deadline_misses = s.deadline_misses.saturating_add(1);
            }
            let steps = st.spread_steps.max(1);
            let parts_per_step = st.num_partitions.div_ceil(steps);
            let end = if steps <= 1 || st.slack_samples == 0 {
                st.num_partitions
            } else {
                parts_per_step.min(st.num_partitions)
            };
            let finish_ifft = end >= st.num_partitions;
            s.in_flight = !finish_ifft;
            s.next_part = end;
            s.samples_since_ready = 0;
            s.steps_done = 1;
            return StageWorkItem::StartBlockAndCmac {
                part_start: 0,
                part_end: end,
                finish_ifft,
            };
        }

        StageWorkItem::None
    }
}
