//! Planning Plane for Non-Uniform Partitioned Convolution (NUPC).
//!
//! Representation-independent (`f32` and fixed-point `Q15` share this exact planner).
//! Validates causal zero-latency stage alignment (`ir_offset >= block_size` for FDL stages),
//! contiguity, FFT block sizes, slack windows, and admission-control budgets before streaming.

use crate::{ConvError, FftSize};

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

/// Conservative upper bound on real multiply-accumulate-equivalent operations that one FDL
/// stage can execute inside a single host callback of `host_block` samples.
///
/// Model (operations, not cycles):
/// - A complex multiply-accumulate (CMAC) costs 4 real MACs; one partition costs `4 * (B + 1)`.
/// - One `N`-point real transform is bounded by `5 * N * log2(N)` (Cooley–Tukey radix-2
///   operation count, conservative for the half-length complex FFT used for real data).
/// - A callback of `H` samples contains at most `t = ceil(H / B)` block triggers, each running
///   one forward RFFT and (unless spread) one IFFT plus all `P` partitions.
/// - A spread stage may additionally finish one in-flight block (one IFFT) and run at most
///   `min(steps, ceil(H / step_interval) + 1)` CMAC sub-steps of `ceil(P / steps)` partitions for
///   each of the `t + 1` blocks overlapping the callback, never more than `(t + 1) * P` partitions.
fn fdl_stage_worst_ops(
    block_size: usize,
    num_partitions: usize,
    slack_samples: usize,
    spread_steps: usize,
    host_block: usize,
) -> Result<usize, ConvError> {
    let overflow = ConvError::ArithmeticOverflow;
    let n = block_size.checked_mul(2).ok_or(overflow)?;
    let log2_n = n.trailing_zeros() as usize;
    let transform_ops = n
        .checked_mul(log2_n)
        .and_then(|v| v.checked_mul(5))
        .ok_or(overflow)?;
    let partition_ops = (block_size + 1).checked_mul(4).ok_or(overflow)?;
    let triggers = host_block.div_ceil(block_size).max(1);

    let spread = spread_steps > 1 && slack_samples > 0;
    let (transforms, partitions) = if !spread {
        (
            triggers.checked_mul(2).ok_or(overflow)?,
            triggers.checked_mul(num_partitions).ok_or(overflow)?,
        )
    } else {
        let blocks = triggers.checked_add(1).ok_or(overflow)?;
        let effective_window = slack_samples.min(block_size - 1).max(1);
        let step_interval = (effective_window / (spread_steps - 1)).max(1);
        let steps_per_block = spread_steps.min(host_block.div_ceil(step_interval) + 1);
        let parts_per_step = num_partitions.div_ceil(spread_steps);
        let by_steps = blocks
            .checked_mul(steps_per_block)
            .and_then(|v| v.checked_mul(parts_per_step))
            .ok_or(overflow)?;
        let by_total = blocks.checked_mul(num_partitions).ok_or(overflow)?;
        (
            triggers.checked_mul(2).ok_or(overflow)? + 1,
            by_steps.min(by_total),
        )
    };

    transforms
        .checked_mul(transform_ops)
        .and_then(|t| {
            partitions
                .checked_mul(partition_ops)
                .and_then(|c| t.checked_add(c))
        })
        .ok_or(overflow)
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

                // Spread partitions across available slack when slack >= host_block_max.
                // Bounded by `num_partitions` and by 8 sub-steps.
                let spread_steps =
                    if slack_samples >= config.host_block_max && config.host_block_max > 0 {
                        let max_slots = (slack_samples / config.host_block_max).max(1);
                        max_slots.min(num_partitions.max(1)).min(8)
                    } else {
                        1
                    };

                worst_ops = worst_ops
                    .checked_add(fdl_stage_worst_ops(
                        st.block_size,
                        num_partitions,
                        slack_samples,
                        spread_steps,
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

    /// Conservative upper-bound estimate of real multiply-accumulate-equivalent operations
    /// (direct MACs, FFT/IFFT operation counts and 4 MACs per complex MAC) that a single host
    /// callback of up to `host_block_max` samples can execute. This is an operation count, not a
    /// cycle count or a WCET proof.
    #[inline]
    pub const fn estimated_worst_ops_per_callback(&self) -> usize {
        self.estimated_worst_ops_per_callback
    }

    /// Evaluates this plan and a computed workspace size in bytes against a [`NonUniformBudget`].
    pub fn check_budget(&self, budget: NonUniformBudget, workspace_bytes: usize) -> BudgetVerdict {
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
            if self.estimated_worst_ops_per_callback > usable {
                return BudgetVerdict::Exceeded;
            }
        }
        if checked_any {
            BudgetVerdict::WithinKnownBounds
        } else {
            BudgetVerdict::Unverified
        }
    }
}
