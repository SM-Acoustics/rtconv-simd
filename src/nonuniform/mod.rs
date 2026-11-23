//! Non-Uniform Partitioned Convolution (NUPC) engine (`#![no_std]`, zero-alloc, zero-latency).
//!
//! Decomposes a causal finite impulse response `h[n]` into ordered regions with increasing
//! partition sizes:
//!
//! `y[n] = sum_{s=0}^{S-1} (x * h_s)[n - D_s]`
//!
//! where each FDL stage satisfies the strict causal zero-latency bound `D_s >= B_s`, and
//! stages with positive slack (`D_s > B_s`) spread their FDL CMAC work deterministically
//! in sample time via [`NonUniformScheduler`].

pub mod memory;
pub mod plan;
pub mod scheduler;
mod stages;

pub use memory::{
    NonUniformCapabilities, NonUniformMemoryRequirements, compute_f32_memory_requirements,
};
pub use plan::{
    BudgetVerdict, MAX_NONUNIFORM_STAGES, NonUniformBudget, NonUniformConfig, NonUniformPlan,
    NonUniformStageConfig, NonUniformStageKind, NonUniformStagePlan,
};
pub use scheduler::{NonUniformScheduler, StageScheduleState, StageWorkItem};

use crate::nonuniform::stages::{
    WorkspaceLayout, direct_stage_process_sample, fdl_stage_cmac_and_maybe_ifft,
    fdl_stage_start_block,
};
use crate::{ConvError, FftBackend, FftSize, SimdBackend};

/// Zero-latency, caller-allocated Non-Uniform Partitioned Convolver (`f32`).
pub struct NonUniformConvolver<'a, F: FftBackend> {
    plan: NonUniformPlan,
    layout: WorkspaceLayout,
    scheduler: NonUniformScheduler,
    workspace: &'a mut [f32],
    backend: SimdBackend,
    fft: F,
    active_ir_len: usize,
    budget_verdict: BudgetVerdict,
}

impl<'a, F: FftBackend> NonUniformConvolver<'a, F> {
    /// Calculates the exact caller-supplied workspace size required for `plan` and `fft`.
    pub fn memory_requirements(
        plan: &NonUniformPlan,
        fft: &F,
    ) -> Result<NonUniformMemoryRequirements, ConvError> {
        compute_f32_memory_requirements(plan, fft)
    }

    /// Prepares a zero-latency [`NonUniformConvolver`] from an impulse response `ir` and a
    /// validated [`NonUniformPlan`] using caller-supplied `workspace` memory.
    ///
    /// Performs all IR partition RFFTs and buffer initialization upfront — zero heap
    /// allocations during preparation or audio processing.
    pub fn prepare(
        ir: &[f32],
        plan: NonUniformPlan,
        workspace: &'a mut [f32],
        backend: SimdBackend,
        fft: F,
    ) -> Result<Self, ConvError> {
        Self::prepare_with_budget(ir, plan, None, workspace, backend, fft)
    }

    /// Prepares a [`NonUniformConvolver`] while enforcing an optional [`NonUniformBudget`]
    /// admission-control contract.
    pub fn prepare_with_budget(
        ir: &[f32],
        plan: NonUniformPlan,
        budget: Option<NonUniformBudget>,
        workspace: &'a mut [f32],
        backend: SimdBackend,
        fft: F,
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
        for (index, &coeff) in ir.iter().enumerate() {
            if !coeff.is_finite() {
                return Err(ConvError::NonFiniteCoefficient { index });
            }
        }

        let req = compute_f32_memory_requirements(&plan, &fft)?;
        let budget_verdict = match budget {
            Some(b) => {
                let verdict = plan.check_budget(b, req.total_bytes);
                if verdict == BudgetVerdict::Exceeded {
                    return Err(ConvError::InvalidNonUniformConfig);
                }
                verdict
            }
            None => BudgetVerdict::Unverified,
        };

        if workspace.len() < req.total_f32 {
            return Err(ConvError::WorkspaceTooSmall {
                required: req.total_f32,
                provided: workspace.len(),
            });
        }

        let layout = WorkspaceLayout::build(&plan, ir.len(), &fft)?;
        workspace[..layout.total_f32].fill(0.0);

        // Pre-load direct coefficients and pre-transform all FDL IR partitions.
        for (idx, st_plan) in plan.stages().iter().enumerate() {
            let st_lay = &layout.stages[idx];
            match st_plan.kind {
                NonUniformStageKind::Direct => {
                    let active = st_lay.direct_active_len;
                    if active > 0 {
                        workspace[st_lay.direct_coeffs_off..st_lay.direct_coeffs_off + active]
                            .copy_from_slice(&ir[st_plan.ir_offset..st_plan.ir_offset + active]);
                    }
                }
                NonUniformStageKind::Fdl => {
                    let b = st_lay.block_size;
                    let n = st_lay.fft_size;
                    let bins = st_lay.bins;
                    let fft_enum = FftSize::from_len(n)?;

                    for p in 0..st_lay.active_partitions {
                        let tap_start = st_plan.ir_offset + p * b;
                        let tap_end = (tap_start + b)
                            .min(st_plan.ir_offset + st_plan.ir_len)
                            .min(ir.len());
                        let copy_len = tap_end.saturating_sub(tap_start);

                        // Zero-pad time_scratch[0..N] and place partition taps in [0..B].
                        let t_off = layout.time_scratch_off;
                        workspace[t_off..t_off + n].fill(0.0);
                        if copy_len > 0 {
                            workspace[t_off..t_off + copy_len]
                                .copy_from_slice(&ir[tap_start..tap_start + copy_len]);
                        }

                        // RFFT partition into ir_re[p], ir_im[p].
                        let re_off = st_lay.ir_re_off + p * bins;
                        let im_off = st_lay.ir_im_off + p * bins;
                        let (before_im, from_im) = workspace.split_at_mut(st_lay.ir_im_off);
                        let (im_region, from_time) =
                            from_im.split_at_mut(layout.time_scratch_off - st_lay.ir_im_off);
                        let (time_slice, fft_sc_slice) = from_time
                            .split_at_mut(layout.fft_scratch_off - layout.time_scratch_off);

                        let out_re = &mut before_im[re_off..re_off + bins];
                        let out_im = &mut im_region
                            [im_off - st_lay.ir_im_off..im_off - st_lay.ir_im_off + bins];
                        fft.rfft_forward(
                            fft_enum,
                            &time_slice[..n],
                            out_re,
                            out_im,
                            &mut fft_sc_slice[..layout.max_fft_scratch],
                        )?;
                    }
                }
            }
        }

        let scheduler = NonUniformScheduler::new(&plan);
        Ok(Self {
            plan,
            layout,
            scheduler,
            workspace,
            backend,
            fft,
            active_ir_len: ir.len(),
            budget_verdict,
        })
    }

    /// Processes an arbitrary-length audio block (`input.len() == output.len()`).
    ///
    /// Supports host blocks smaller than, equal to, or larger than internal FFT block sizes,
    /// as well as dynamic block-size changes across callbacks.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), ConvError> {
        if input.len() != output.len() {
            return Err(ConvError::MismatchedBufferLength {
                expected: input.len(),
                actual: output.len(),
            });
        }
        for (&x, y_out) in input.iter().zip(output.iter_mut()) {
            *y_out = self.process_sample_inner(x)?;
        }
        Ok(())
    }

    /// Processes a single audio sample with zero algorithmic latency.
    pub fn process_sample(&mut self, x: f32) -> Result<f32, ConvError> {
        self.process_sample_inner(x)
    }

    #[inline]
    fn process_sample_inner(&mut self, x: f32) -> Result<f32, ConvError> {
        let mut sum = 0.0f32;
        let stage_count = self.layout.stage_count;

        for idx in 0..stage_count {
            if self.layout.stages[idx].kind == 0 {
                let st = &mut self.layout.stages[idx];
                sum += direct_stage_process_sample(self.workspace, st, x, self.backend);
            } else {
                let st = &mut self.layout.stages[idx];
                let b = st.block_size;
                let fill_before = self.scheduler.stage_state(idx).input_fill;
                // Write incoming sample into overlap-save second half (`overlap_in[B + fill]`).
                self.workspace[st.overlap_in_off + b + fill_before] = x;

                // Read causally aligned delayed contribution for current sample.
                let read_pos = self.scheduler.stage_state(idx).delay_read_pos;
                let ring_idx = read_pos % st.delay_ring_len;
                let y_stage = self.workspace[st.delay_ring_off + ring_idx];
                self.workspace[st.delay_ring_off + ring_idx] = 0.0;
                self.scheduler.stage_state_mut(idx).delay_read_pos = read_pos + 1;
                sum += y_stage;

                // Advance sample-time scheduler and execute any FDL work due at this boundary.
                let work = self.scheduler.on_fdl_sample(idx, &self.plan);
                match work {
                    StageWorkItem::None => {}
                    StageWorkItem::StartBlockAndCmac {
                        part_start,
                        part_end,
                        finish_ifft,
                    } => {
                        let mut write_pos = self.scheduler.stage_state(idx).delay_write_pos;
                        fdl_stage_start_block(
                            self.workspace,
                            st,
                            self.layout.time_scratch_off,
                            self.layout.fft_scratch_off,
                            self.layout.max_fft_scratch,
                            part_start,
                            part_end,
                            finish_ifft,
                            &mut write_pos,
                            self.backend,
                            &self.fft,
                        )?;
                        self.scheduler.stage_state_mut(idx).delay_write_pos = write_pos;
                    }
                    StageWorkItem::ContinueCmac {
                        part_start,
                        part_end,
                        finish_ifft,
                    } => {
                        let mut write_pos = self.scheduler.stage_state(idx).delay_write_pos;
                        fdl_stage_cmac_and_maybe_ifft(
                            self.workspace,
                            st,
                            self.layout.time_scratch_off,
                            self.layout.fft_scratch_off,
                            self.layout.max_fft_scratch,
                            part_start,
                            part_end,
                            finish_ifft,
                            &mut write_pos,
                            self.backend,
                            &self.fft,
                        )?;
                        self.scheduler.stage_state_mut(idx).delay_write_pos = write_pos;
                    }
                }
            }
        }

        self.scheduler.advance_sample();
        Ok(sum)
    }

    /// Deterministically resets all delay lines, FDL rings, and scheduler cursors while
    /// preserving the pre-transformed impulse response spectra.
    pub fn reset(&mut self) {
        for idx in 0..self.layout.stage_count {
            let st = &mut self.layout.stages[idx];
            if st.kind == 0 {
                self.workspace[st.direct_hist_off..st.direct_hist_off + st.direct_len].fill(0.0);
                st.direct_head = 0;
            } else {
                let part_bins = st.num_partitions * st.bins;
                self.workspace[st.overlap_in_off..st.overlap_in_off + st.fft_size].fill(0.0);
                self.workspace[st.fdl_re_off..st.fdl_re_off + part_bins].fill(0.0);
                self.workspace[st.fdl_im_off..st.fdl_im_off + part_bins].fill(0.0);
                self.workspace[st.acc_re_off..st.acc_re_off + st.bins].fill(0.0);
                self.workspace[st.acc_im_off..st.acc_im_off + st.bins].fill(0.0);
                self.workspace[st.delay_ring_off..st.delay_ring_off + st.delay_ring_len].fill(0.0);
                st.fdl_head = 0;
            }
        }
        self.scheduler.reset(&self.plan);
    }

    /// Algorithmic latency in samples (always `0` for valid causal NUPC plans).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        0
    }

    /// Total deadline misses detected by the sample-time scheduler (always `0` for valid plans).
    #[inline]
    pub fn deadline_misses(&self) -> u64 {
        self.scheduler.total_deadline_misses()
    }

    /// Active validated [`NonUniformPlan`].
    #[inline]
    pub fn plan(&self) -> &NonUniformPlan {
        &self.plan
    }

    /// Runtime capabilities and backend metadata.
    pub fn capabilities(&self) -> NonUniformCapabilities {
        NonUniformCapabilities {
            stage_count: self.plan.stage_count(),
            simd_backend: self.backend.name(),
            simd_active: self.backend.is_simd_active(),
            fft_backend: self.fft.name(),
            fft_is_simd: self.fft.is_simd_accelerated(),
            max_ir_len: self.plan.max_ir_len(),
            active_ir_len: self.active_ir_len,
            latency_samples: 0,
            budget_verdict: self.budget_verdict,
        }
    }
}
