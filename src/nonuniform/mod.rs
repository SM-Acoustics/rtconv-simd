//! Non-Uniform Partitioned Convolution (NUPC) engine (`#![no_std]`, zero-alloc, zero-latency).
//!
//! Decomposes a causal finite impulse response `h[n]` into ordered regions with increasing
//! partition sizes:
//!
//! `y[n] = sum_{s=0}^{S-1} (x * h_s)[n - D_s]`
//!
//! where each FDL stage satisfies the strict causal zero-latency bound `D_s >= B_s`, and
//! stages with positive slack (`D_s > B_s`) distribute forward RFFT stages, spectral CMAC
//! partitions, and inverse IFFT stages deterministically in sample time via
//! [`NonUniformScheduler`], with deferred execution shrinking each stage's output delay ring
//! to power-of-two `2 * B_s`.

pub mod memory;
pub mod optimizer;
pub mod plan;
pub mod scheduler;
mod stages;

pub use memory::{
    NonUniformCapabilities, NonUniformMemoryRequirements, compute_f32_memory_requirements,
};
pub use optimizer::{
    BlockModePartition, BlockModePolicy, BlockModeSegment, ClearancePolicy, CostModel,
    MAX_OPTIMIZER_FDL_STAGES, OPTIMIZER_BLOCK_SIZES, OptimizationReport, OptimizerScratchCell,
    block_mode_scratch_len, optimize_block_mode, optimizer_scratch_len, preset_esp32_c6_q15,
    preset_esp32_p4_f32,
};
pub use plan::{
    BudgetVerdict, MAX_NONUNIFORM_STAGES, NonUniformBudget, NonUniformConfig, NonUniformPlan,
    NonUniformStageConfig, NonUniformStageKind, NonUniformStagePlan,
};
pub use scheduler::{NonUniformScheduler, StageScheduleState, StageWorkItem, stage_work_item_ops};

use crate::nonuniform::stages::{
    WorkspaceLayout, direct_stage_process_sample, fdl_stage_run_cmac, fdl_stage_run_ifft_steps,
    fdl_stage_run_rfft_steps,
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
                let verdict = plan.check_budget_for_backend(b, req.total_bytes, &fft)?;
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
                        crate::simd_kernels::copy_flush_tiny(
                            &mut workspace
                                [st_lay.direct_coeffs_off..st_lay.direct_coeffs_off + active],
                            &ir[st_plan.ir_offset..st_plan.ir_offset + active],
                        );
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
                            crate::simd_kernels::copy_flush_tiny(
                                &mut workspace[t_off..t_off + copy_len],
                                &ir[tap_start..tap_start + copy_len],
                            );
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
                        crate::simd_kernels::flush_tiny_in_place(out_re);
                        crate::simd_kernels::flush_tiny_in_place(out_im);
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
        let x = crate::simd_kernels::flush_tiny(x);
        let shared_in_off = self.layout.shared_in_off;
        let shared_in_len = self.layout.shared_in_len;
        let in_write_pos = self.layout.shared_in_write_pos;
        if shared_in_len > 0 {
            self.workspace[shared_in_off + in_write_pos] = x;
        }

        let mut sum = 0.0f32;
        let stage_count = self.layout.stage_count;
        let time_scratch_off = self.layout.time_scratch_off;
        let fft_scratch_off = self.layout.fft_scratch_off;
        let max_fft_scratch = self.layout.max_fft_scratch;

        for idx in 0..stage_count {
            if self.layout.stages[idx].kind == 0 {
                let st = &mut self.layout.stages[idx];
                sum += direct_stage_process_sample(self.workspace, st, x, self.backend);
            } else {
                let st = &mut self.layout.stages[idx];

                // Read causally aligned delayed contribution (power-of-two bitmask, zero division).
                let read_pos = self.scheduler.stage_state(idx).delay_read_pos;
                let ring_idx = read_pos & st.delay_ring_mask;
                let y_stage = self.workspace[st.delay_ring_off + ring_idx];
                self.workspace[st.delay_ring_off + ring_idx] = 0.0;
                self.scheduler.stage_state_mut(idx).delay_read_pos = read_pos + 1;
                sum += y_stage;

                // Advance sample-time scheduler and execute any FDL work due at this boundary.
                let work = self.scheduler.on_fdl_sample_with_steps(
                    idx,
                    &self.plan,
                    st.rfft_steps,
                    st.ifft_steps,
                );
                match work {
                    StageWorkItem::None => {}
                    StageWorkItem::RfftSteps {
                        step_start,
                        step_end,
                        finish_rfft,
                    } => {
                        fdl_stage_run_rfft_steps(
                            self.workspace,
                            st,
                            shared_in_off,
                            shared_in_len,
                            in_write_pos,
                            time_scratch_off,
                            fft_scratch_off,
                            max_fft_scratch,
                            step_start,
                            step_end,
                            finish_rfft,
                            &self.fft,
                        )?;
                    }
                    StageWorkItem::ContinueCmac {
                        part_start,
                        part_end,
                        finish_ifft,
                    } => {
                        fdl_stage_run_cmac(self.workspace, st, part_start, part_end, self.backend);
                        if finish_ifft {
                            let mut write_pos = self.scheduler.stage_state(idx).delay_write_pos;
                            fdl_stage_run_ifft_steps(
                                self.workspace,
                                st,
                                time_scratch_off,
                                fft_scratch_off,
                                max_fft_scratch,
                                0,
                                st.ifft_steps,
                                true,
                                &mut write_pos,
                                &self.fft,
                            )?;
                            self.scheduler.stage_state_mut(idx).delay_write_pos = write_pos;
                        }
                    }
                    StageWorkItem::IfftSteps {
                        step_start,
                        step_end,
                        commit_output,
                    } => {
                        let mut write_pos = self.scheduler.stage_state(idx).delay_write_pos;
                        fdl_stage_run_ifft_steps(
                            self.workspace,
                            st,
                            time_scratch_off,
                            fft_scratch_off,
                            max_fft_scratch,
                            step_start,
                            step_end,
                            commit_output,
                            &mut write_pos,
                            &self.fft,
                        )?;
                        self.scheduler.stage_state_mut(idx).delay_write_pos = write_pos;
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
                            fdl_stage_run_rfft_steps(
                                self.workspace,
                                st,
                                shared_in_off,
                                shared_in_len,
                                in_write_pos,
                                time_scratch_off,
                                fft_scratch_off,
                                max_fft_scratch,
                                rfft_start,
                                rfft_end,
                                finish_rfft,
                                &self.fft,
                            )?;
                        }
                        if cmac_start < cmac_end {
                            fdl_stage_run_cmac(
                                self.workspace,
                                st,
                                cmac_start,
                                cmac_end,
                                self.backend,
                            );
                        }
                        if ifft_start < ifft_end {
                            let mut write_pos = self.scheduler.stage_state(idx).delay_write_pos;
                            fdl_stage_run_ifft_steps(
                                self.workspace,
                                st,
                                time_scratch_off,
                                fft_scratch_off,
                                max_fft_scratch,
                                ifft_start,
                                ifft_end,
                                commit_output,
                                &mut write_pos,
                                &self.fft,
                            )?;
                            self.scheduler.stage_state_mut(idx).delay_write_pos = write_pos;
                        }
                    }
                }
            }
        }

        if shared_in_len > 0 {
            let next = in_write_pos + 1;
            self.layout.shared_in_write_pos = if next == shared_in_len { 0 } else { next };
        }
        self.scheduler.advance_sample();
        Ok(crate::simd_kernels::flush_tiny(sum))
    }

    /// Deterministically resets all delay lines, FDL rings, shared input history, and scheduler
    /// cursors while preserving the pre-transformed impulse response spectra.
    pub fn reset(&mut self) {
        if self.layout.shared_in_len > 0 {
            self.workspace
                [self.layout.shared_in_off..self.layout.shared_in_off + self.layout.shared_in_len]
                .fill(0.0);
            self.layout.shared_in_write_pos = 0;
        }
        for idx in 0..self.layout.stage_count {
            let st = &mut self.layout.stages[idx];
            if st.kind == 0 {
                self.workspace[st.direct_hist_off..st.direct_hist_off + st.direct_len].fill(0.0);
                st.direct_head = 0;
            } else {
                let part_bins = st.num_partitions * st.bins;
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
