//! Planning Plane for Non-Uniform Partitioned Convolution (NUPC).
//!
//! Representation-independent (`f32` and fixed-point `Q15` share this exact planner).
//! Validates causal zero-latency stage alignment (`ir_offset >= block_size` for FDL stages),
//! contiguity, FFT block sizes, slack windows, and admission-control budgets before streaming.

use crate::{ConvError, FftBackend, FftSize};

/// Maximum number of stages supported in a single [`NonUniformPlan`] (stack-allocated, zero heap).
pub const MAX_NONUNIFORM_STAGES: usize = 8;

/// Algorithm executed by a single NUPC stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonUniformStageKind {
    /// Time-domain direct FIR convolution (must start at `ir_offset == 0`, `block_size == 1`).
    Direct,
    /// Frequency-Domain Delay Line (50% overlap-save uniform partitions of length `block_size`).
    Fdl,
}

/// Caller-provided description of one impulse-response region in a NUPC plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonUniformStageConfig {
    /// Starting tap index `D_s` in the full impulse response `h[n]`.
    pub ir_offset: usize,
    /// Number of taps `L_s` covered by this stage.
    pub ir_len: usize,
    /// Partition block size `B_s`: `1` for Direct FIR head, or a supported power-of-two FDL
    /// block size in `32..=8192` (corresponding to FFT sizes `N64` through `N16384`).
    /// Note that fixed-point Q1.15 engines accept FDL block sizes up to `1024` (`N2048`).
    pub block_size: usize,
}

/// Top-level configuration for validating and building a [`NonUniformPlan`].
#[derive(Clone, Copy, Debug)]
pub struct NonUniformConfig<'a> {
    /// Maximum impulse response length supported by this plan.
    pub max_ir_len: usize,
    /// Maximum number of samples processed per host callback (used for budget and spread planning).
    pub host_block_max: usize,
    /// Contiguous, ordered list of IR stages covering `0..total_ir_len`.
    pub stages: &'a [NonUniformStageConfig],
}

/// Admission-control budget contract evaluated prior to streaming.
///
/// Estimates are deterministic upper bounds on memory and operation counts, **not**
/// formal hardware WCET proofs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonUniformBudget {
    /// Audio sample rate in Hz (e.g. `48_000`).
    pub sample_rate_hz: u32,
    /// Maximum frames processed per audio callback.
    pub max_frames_per_callback: usize,
    /// Nominal callback deadline in microseconds (informational; not a hardware WCET proof).
    pub deadline_us: u32,
    /// Maximum workspace memory in bytes permitted by the integrator (`0` = unchecked).
    pub max_memory_bytes: usize,
    /// Maximum estimated MAC/CMAC operations permitted per callback (`0` = unchecked).
    pub max_ops_per_callback: usize,
    /// Integrator-defined safety margin percentage (`0..=90`).
    pub safety_margin_percent: u8,
}

/// Result of evaluating a [`NonUniformPlan`] against an optional [`NonUniformBudget`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetVerdict {
    /// All checked memory and operation bounds are within the declared budget.
    WithinKnownBounds,
    /// One or more checked limits exceed the declared budget (configuration rejected).
    Exceeded,
    /// No hard budget bounds were supplied or timing cannot be proved without hardware measurement.
    Unverified,
}

/// Validated, immutable description of a single NUPC stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonUniformStagePlan {
    /// Direct FIR or FDL stage.
    pub kind: NonUniformStageKind,
    /// Starting tap index `D_s` in `h[n]`.
    pub ir_offset: usize,
    /// Maximum tap count `L_s` allocated for this stage.
    pub ir_len: usize,
    /// Block size `B_s` (`1` for Direct, power-of-two `32..=8192` for FDL).
    pub block_size: usize,
    /// FFT size `N_s = 2 * B_s` (`0` for Direct).
    pub fft_size: usize,
    /// Number of uniform partitions `ceil(L_s / B_s)` (`0` for Direct).
    pub num_partitions: usize,
    /// Causal slack `D_s - B_s` in samples (`0` for Direct).
    /// Output produced at `t_ready = k * B_s` is read at `t_due = k * B_s + slack_samples`.
    pub slack_samples: usize,
    /// Number of spread sub-steps across which FDL CMAC work can be distributed (`>= 1`).
    pub spread_steps: usize,
}

impl NonUniformStagePlan {
    const fn empty() -> Self {
        Self {
            kind: NonUniformStageKind::Direct,
            ir_offset: 0,
            ir_len: 0,
            block_size: 1,
            fft_size: 0,
            num_partitions: 0,
            slack_samples: 0,
            spread_steps: 1,
        }
    }
}

/// Validated, stack-allocated NUPC execution plan shared by `f32` and fixed-point engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonUniformPlan {
    stages: [NonUniformStagePlan; MAX_NONUNIFORM_STAGES],
    stage_count: usize,
    max_ir_len: usize,
    host_block_max: usize,
    estimated_worst_ops_per_callback: usize,
}

/// Operation count model constants for one FDL stage of block size `B` (`N = 2 * B`):
/// - One `N`-point real transform is bounded by `5 * N * log2(N)`.
/// - One CMAC partition costs `4 * (B + 1)` real multiply-accumulate equivalents.
#[inline]
pub(crate) fn fdl_stage_op_weights(
    block_size: usize,
    rfft_steps: usize,
    ifft_steps: usize,
) -> Result<(usize, usize, usize), ConvError> {
    let overflow = ConvError::ArithmeticOverflow;
    let n = block_size.checked_mul(2).ok_or(overflow)?;
    let log2_n = n.trailing_zeros() as usize;
    let transform_ops = n
        .checked_mul(log2_n)
        .and_then(|v| v.checked_mul(5))
        .ok_or(overflow)?;
    let r = rfft_steps.max(1);
    let i = ifft_steps.max(1);
    let rfft_step_ops = transform_ops.div_ceil(r);
    let ifft_step_ops = transform_ops.div_ceil(i);
    let partition_ops = block_size
        .checked_add(1)
        .and_then(|v| v.checked_mul(4))
        .ok_or(overflow)?;
    Ok((rfft_step_ops, ifft_step_ops, partition_ops))
}

/// Exact, alignment-independent upper bound on real multiply-accumulate-equivalent operations
/// that one time-distributed FDL stage can execute inside ANY contiguous `host_block`-sample
/// callback under the periodic floor schedule `target(slot) = ((slot + 1) * U) / span`.
///
/// Proof of soundness (all `H >= 1`, `B >= 1`, `1 <= span <= B`, `U = R + P + I`):
/// - Write `H = q * B + rem` (`q = H / B`, `rem = H % B`).
/// - Every `B` consecutive samples execute exactly `U` units (`full_block_ops`).
/// - Over the remaining `rem` samples (whether inside one active window or straddling the tail
///   of block `k` and the head of block `k + 1`), the schedule executes at most
///   `u_rem = min(U, ceil(rem * U / span))` units because
///   `ceil(L_1 * U / span) + floor(L_2 * U / span) <= ceil((L_1 + L_2) * U / span)`.
/// - Assigning those `u_rem` units to the highest-cost unit types first (RFFT/IFFT steps before
///   CMAC partitions) produces an airtight upper bound on actual callback operations.
pub(crate) fn fdl_stage_worst_ops_for_steps(
    block_size: usize,
    num_partitions: usize,
    slack_samples: usize,
    host_block: usize,
    rfft_steps: usize,
    ifft_steps: usize,
) -> Result<usize, ConvError> {
    let overflow = ConvError::ArithmeticOverflow;
    let r = rfft_steps.max(1);
    let i = ifft_steps.max(1);
    let (rfft_step_ops, ifft_step_ops, partition_ops) = fdl_stage_op_weights(block_size, r, i)?;

    let total_units = r
        .checked_add(num_partitions)
        .and_then(|v| v.checked_add(i))
        .ok_or(overflow)?;
    let full_block_ops = r
        .checked_mul(rfft_step_ops)
        .and_then(|v| v.checked_add(i.checked_mul(ifft_step_ops)?))
        .and_then(|v| v.checked_add(num_partitions.checked_mul(partition_ops)?))
        .ok_or(overflow)?;

    let w_eff = slack_samples.min(block_size.saturating_sub(1));
    let span = w_eff.checked_add(1).ok_or(overflow)?;

    let full_periods = host_block / block_size;
    let rem = host_block % block_size;

    let rem_ops = if rem == 0 {
        0
    } else {
        let u_rem = rem
            .checked_mul(total_units)
            .ok_or(overflow)?
            .div_ceil(span)
            .min(total_units);
        let (hi_ops, hi_cnt, lo_ops, lo_cnt) = if rfft_step_ops >= ifft_step_ops {
            (rfft_step_ops, r, ifft_step_ops, i)
        } else {
            (ifft_step_ops, i, rfft_step_ops, r)
        };
        let take_hi = u_rem.min(hi_cnt);
        let after_hi = u_rem - take_hi;
        let take_lo = after_hi.min(lo_cnt);
        let take_cmac = after_hi - take_lo;
        take_hi
            .checked_mul(hi_ops)
            .and_then(|v| v.checked_add(take_lo.checked_mul(lo_ops)?))
            .and_then(|v| v.checked_add(take_cmac.checked_mul(partition_ops)?))
            .ok_or(overflow)?
    };

    full_periods
        .checked_mul(full_block_ops)
        .and_then(|v| v.checked_add(rem_ops))
        .ok_or(overflow)
}

/// Conservative time-distributed upper bound across both native staged engines:
/// `f32` (`SimdFixedFftBackend`: `log2(B)` steps) and `Q1.15` (`fft_q15`: `log2(2*B)` steps).
fn fdl_stage_worst_ops(
    block_size: usize,
    num_partitions: usize,
    slack_samples: usize,
    host_block: usize,
) -> Result<usize, ConvError> {
    let f32_steps = block_size.trailing_zeros() as usize;
    let q15_steps = f32_steps + 1;
    let ops_f32 = fdl_stage_worst_ops_for_steps(
        block_size,
        num_partitions,
        slack_samples,
        host_block,
        f32_steps,
        f32_steps,
    )?;
    let ops_q15 = fdl_stage_worst_ops_for_steps(
        block_size,
        num_partitions,
        slack_samples,
        host_block,
        q15_steps,
        q15_steps,
    )?;
    Ok(ops_f32.max(ops_q15))
}

impl NonUniformPlan {
    /// Validates a caller-provided [`NonUniformConfig`] and builds a [`NonUniformPlan`].
    ///
    /// Enforces:
    /// - `1..=MAX_NONUNIFORM_STAGES` stages, `max_ir_len > 0`, `host_block_max > 0`.
    /// - Contiguous coverage starting at `ir_offset == 0` with zero gaps or overlaps.
    /// - Strict zero-latency causality: every FDL stage requires `ir_offset >= block_size`.
    /// - Supported FDL block sizes: powers of two in `32..=8192` (`N64` through `N16384`).
    pub fn validate(config: NonUniformConfig<'_>) -> Result<Self, ConvError> {
        if config.max_ir_len == 0
            || config.host_block_max == 0
            || config.stages.is_empty()
            || config.stages.len() > MAX_NONUNIFORM_STAGES
        {
            return Err(ConvError::InvalidNonUniformConfig);
        }

        let mut stages = [NonUniformStagePlan::empty(); MAX_NONUNIFORM_STAGES];
        let mut cursor = 0usize;
        let mut worst_ops = 0usize;

        for (idx, st) in config.stages.iter().enumerate() {
            if st.ir_len == 0 || st.ir_offset != cursor {
                return Err(ConvError::InvalidNonUniformConfig);
            }
            let end = cursor
                .checked_add(st.ir_len)
                .ok_or(ConvError::ArithmeticOverflow)?;
            if end > config.max_ir_len {
                return Err(ConvError::InvalidNonUniformConfig);
            }

            let plan = if st.block_size == 1 {
                if st.ir_offset != 0 || idx != 0 {
                    return Err(ConvError::InvalidNonUniformConfig);
                }
                let ops = st
                    .ir_len
                    .checked_mul(config.host_block_max)
                    .ok_or(ConvError::ArithmeticOverflow)?;
                worst_ops = worst_ops
                    .checked_add(ops)
                    .ok_or(ConvError::ArithmeticOverflow)?;
                NonUniformStagePlan {
                    kind: NonUniformStageKind::Direct,
                    ir_offset: 0,
                    ir_len: st.ir_len,
                    block_size: 1,
                    fft_size: 0,
                    num_partitions: 0,
                    slack_samples: 0,
                    spread_steps: 1,
                }
            } else {
                let fft = FftSize::from_block_size(st.block_size)?;
                // Strict causal zero-latency condition: D_s >= B_s.
                if st.ir_offset < st.block_size {
                    return Err(ConvError::InvalidNonUniformConfig);
                }
                let fft_size = fft.len();
                let num_partitions = st.ir_len.div_ceil(st.block_size);
                let slack_samples = st.ir_offset - st.block_size;

                // Active time-distributed work slots per block across `W_eff + 1` samples
                // (`1` when `slack == 0`, up to `min(2 * log2(B) + P, W_eff + 1)` otherwise).
                let w_eff = slack_samples.min(st.block_size.saturating_sub(1));
                let span = w_eff + 1;
                let r_steps = st.block_size.trailing_zeros() as usize;
                let total_units = 2 * r_steps + num_partitions;
                let spread_steps = if slack_samples > 0 {
                    total_units.min(span).max(1)
                } else {
                    1
                };

                worst_ops = worst_ops
                    .checked_add(fdl_stage_worst_ops(
                        st.block_size,
                        num_partitions,
                        slack_samples,
                        config.host_block_max,
                    )?)
                    .ok_or(ConvError::ArithmeticOverflow)?;

                NonUniformStagePlan {
                    kind: NonUniformStageKind::Fdl,
                    ir_offset: st.ir_offset,
                    ir_len: st.ir_len,
                    block_size: st.block_size,
                    fft_size,
                    num_partitions,
                    slack_samples,
                    spread_steps,
                }
            };

            stages[idx] = plan;
            cursor = end;
        }

        if cursor == 0 {
            return Err(ConvError::InvalidNonUniformConfig);
        }

        Ok(Self {
            stages,
            stage_count: config.stages.len(),
            max_ir_len: config.max_ir_len,
            host_block_max: config.host_block_max,
            estimated_worst_ops_per_callback: worst_ops,
        })
    }

    /// Builds the recommended multi-stage NUPC profile using the crate's verified FFT sizes
    /// (`Direct 0..32`, `FDL B=32 / N64 32..256`, `FDL B=64 / N128 256..1024`,
    /// `FDL B=128 / N256 1024..max_ir_len`), clipped cleanly to `max_ir_len`.
    pub fn default_profile(max_ir_len: usize, host_block_max: usize) -> Result<Self, ConvError> {
        if max_ir_len == 0 || host_block_max == 0 {
            return Err(ConvError::InvalidNonUniformConfig);
        }
        let mut cfgs = [NonUniformStageConfig {
            ir_offset: 0,
            ir_len: 0,
            block_size: 1,
        }; 4];
        let mut count = 0usize;

        let head = max_ir_len.min(32);
        cfgs[count] = NonUniformStageConfig {
            ir_offset: 0,
            ir_len: head,
            block_size: 1,
        };
        count += 1;

        if max_ir_len > 32 {
            let end = max_ir_len.min(256);
            cfgs[count] = NonUniformStageConfig {
                ir_offset: 32,
                ir_len: end - 32,
                block_size: 32,
            };
            count += 1;
        }
        if max_ir_len > 256 {
            let end = max_ir_len.min(1024);
            cfgs[count] = NonUniformStageConfig {
                ir_offset: 256,
                ir_len: end - 256,
                block_size: 64,
            };
            count += 1;
        }
        if max_ir_len > 1024 {
            cfgs[count] = NonUniformStageConfig {
                ir_offset: 1024,
                ir_len: max_ir_len - 1024,
                block_size: 128,
            };
            count += 1;
        }

        Self::validate(NonUniformConfig {
            max_ir_len,
            host_block_max,
            stages: &cfgs[..count],
        })
    }

    /// Validated stages in ascending `ir_offset` order.
    #[inline]
    pub fn stages(&self) -> &[NonUniformStagePlan] {
        &self.stages[..self.stage_count]
    }

    /// Number of active stages in the plan.
    #[inline]
    pub const fn stage_count(&self) -> usize {
        self.stage_count
    }

    /// Configured maximum impulse response length.
    #[inline]
    pub const fn max_ir_len(&self) -> usize {
        self.max_ir_len
    }

    /// Maximum host callback block size planned for.
    #[inline]
    pub const fn host_block_max(&self) -> usize {
        self.host_block_max
    }

    /// Direct FIR multiply-accumulate operations executed per input sample (`0` if no direct stage).
    #[inline]
    pub const fn direct_sample_ops(&self) -> usize {
        if self.stage_count > 0 && matches!(self.stages[0].kind, NonUniformStageKind::Direct) {
            self.stages[0].ir_len
        } else {
            0
        }
    }

    /// Conservative upper-bound estimate of real multiply-accumulate-equivalent operations
    /// (direct MACs, time-distributed FFT/IFFT stage operations and 4 MACs per complex MAC)
    /// that a single host callback of up to `host_block_max` samples can execute on the native
    /// staged engines ([`crate::SimdFixedFftBackend`] and `FixedQ15NonUniformConvolver`).
    /// This is an operation count, not a cycle count or a WCET proof.
    #[inline]
    pub const fn estimated_worst_ops_per_callback(&self) -> usize {
        self.estimated_worst_ops_per_callback
    }

    /// Exact alignment-independent worst-case per-callback operation bound for a specific
    /// [`FftBackend`] (reflecting its actual [`FftBackend::staged_forward_steps`] and
    /// [`FftBackend::staged_inverse_steps`], including `1` for monolithic backends such as
    /// [`crate::MicroFftBackend`]).
    pub fn estimated_worst_ops_for_backend<F: FftBackend>(
        &self,
        fft: &F,
    ) -> Result<usize, ConvError> {
        let mut worst_ops = 0usize;
        for st in self.stages() {
            match st.kind {
                NonUniformStageKind::Direct => {
                    let ops = st
                        .ir_len
                        .checked_mul(self.host_block_max)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    worst_ops = worst_ops
                        .checked_add(ops)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                }
                NonUniformStageKind::Fdl => {
                    let fft_enum = FftSize::from_len(st.fft_size)?;
                    let r_steps = fft.staged_forward_steps(fft_enum);
                    let i_steps = fft.staged_inverse_steps(fft_enum);
                    let stage_ops = fdl_stage_worst_ops_for_steps(
                        st.block_size,
                        st.num_partitions,
                        st.slack_samples,
                        self.host_block_max,
                        r_steps,
                        i_steps,
                    )?;
                    worst_ops = worst_ops
                        .checked_add(stage_ops)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                }
            }
        }
        Ok(worst_ops)
    }

    fn check_budget_with_ops(
        &self,
        budget: NonUniformBudget,
        workspace_bytes: usize,
        worst_ops: usize,
    ) -> BudgetVerdict {
        if budget.safety_margin_percent > 90
            || budget.sample_rate_hz == 0
            || budget.max_frames_per_callback == 0
        {
            return BudgetVerdict::Exceeded;
        }
        if self.host_block_max > budget.max_frames_per_callback {
            return BudgetVerdict::Exceeded;
        }
        let mut checked_any = false;
        if budget.max_memory_bytes > 0 {
            checked_any = true;
            if workspace_bytes > budget.max_memory_bytes {
                return BudgetVerdict::Exceeded;
            }
        }
        if budget.max_ops_per_callback > 0 {
            checked_any = true;
            let usable = budget
                .max_ops_per_callback
                .saturating_mul(100 - budget.safety_margin_percent as usize)
                / 100;
            if worst_ops > usable {
                return BudgetVerdict::Exceeded;
            }
        }
        if checked_any {
            BudgetVerdict::WithinKnownBounds
        } else {
            BudgetVerdict::Unverified
        }
    }

    /// Evaluates this plan and a computed workspace size in bytes against a [`NonUniformBudget`].
    pub fn check_budget(&self, budget: NonUniformBudget, workspace_bytes: usize) -> BudgetVerdict {
        self.check_budget_with_ops(
            budget,
            workspace_bytes,
            self.estimated_worst_ops_per_callback,
        )
    }

    /// Evaluates this plan against a [`NonUniformBudget`] using the exact worst-case operation
    /// bound for `fft` (accounting for monolithic vs. time-distributed transform backends).
    pub fn check_budget_for_backend<F: FftBackend>(
        &self,
        budget: NonUniformBudget,
        workspace_bytes: usize,
        fft: &F,
    ) -> Result<BudgetVerdict, ConvError> {
        let worst_ops = self.estimated_worst_ops_for_backend(fft)?;
        Ok(self.check_budget_with_ops(budget, workspace_bytes, worst_ops))
    }
}
