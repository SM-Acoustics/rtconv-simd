//! Memory requirements and runtime capabilities for Non-Uniform Partitioned Convolution (NUPC).

use crate::nonuniform::plan::{BudgetVerdict, NonUniformPlan, NonUniformStageKind};
use crate::{ConvError, FftBackend, FftSize};

/// Explicit memory breakdown for [`crate::NonUniformConvolver`] in `f32` elements and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonUniformMemoryRequirements {
    /// Time-domain state buffers (direct FIR history + coefficients, plus the single shared
    /// real input history ring for all FDL stages).
    pub state_f32: usize,
    /// Frequency-domain buffers (FDL ring spectra + pre-transformed IR partition spectra + accumulators).
    pub spectral_f32: usize,
    /// Per-stage power-of-two causal output delay ring buffers (`2 * B_s` per FDL stage).
    pub delay_f32: usize,
    /// Shared FFT/IFFT scratch memory (sized for the largest FFT stage in the plan).
    pub scratch_f32: usize,
    /// Total required `&mut [f32]` workspace elements.
    pub total_f32: usize,
    /// Total required workspace size in bytes (`total_f32 * size_of::<f32>()`).
    pub total_bytes: usize,
}

/// Runtime metadata describing an active [`crate::NonUniformConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonUniformCapabilities {
    /// Number of active stages in the plan.
    pub stage_count: usize,
    /// Selected SIMD backend label.
    pub simd_backend: &'static str,
    /// Whether vector SIMD instructions are active for FIR/CMAC kernels.
    pub simd_active: bool,
    /// Selected FFT backend label.
    pub fft_backend: &'static str,
    /// Whether the FFT/IFFT transform itself is SIMD-vectorized.
    pub fft_is_simd: bool,
    /// Maximum configured impulse response length.
    pub max_ir_len: usize,
    /// Active impulse response length.
    pub active_ir_len: usize,
    /// Algorithmic latency in samples (always `0` for valid causal NUPC plans).
    pub latency_samples: usize,
    /// Admission-control budget verdict (`Unverified` if no budget was supplied).
    pub budget_verdict: BudgetVerdict,
}

/// Computes the power-of-two output ring capacity for an FDL stage: `2 * block_size`.
///
/// With deferred scheduling (`defer_s = slack_s.saturating_sub(block_size - 1)`), block `k`
/// starts at `t_start(k) = (k + 1) * B_s - 1 + defer_s` and commits `B_s` samples at
/// `[k * B_s + D_s .. (k + 1) * B_s + D_s)` on or before `t_fin(k) = k * B_s + D_s - 1`.
/// Even if block `k` commits immediately at `t_start(k)`, the live unread + written span is
/// `<= B_s + min(slack_s, B_s - 1) <= 2 * B_s - 1 < 2 * B_s`, guaranteeing zero overwrite
/// with branchless bitmask indexing (`pos & (2 * B_s - 1)`).
#[inline]
pub(crate) fn stage_delay_ring_len(block_size: usize) -> Result<usize, ConvError> {
    block_size
        .checked_mul(2)
        .ok_or(ConvError::ArithmeticOverflow)
}

/// Computes the exact capacity of the single shared real input history ring for all FDL stages:
/// `0` if the plan has no FDL stages, otherwise `max_s(max(2 * B_s, D_s + 1))`.
///
/// At `t_start(k) = (k + 1) * B_s - 1 + defer_s`, stage `s` captures the `2 * B_s` overlap-save
/// window ending at `x[(k + 1) * B_s - 1]`. The oldest sample needed has age
/// `2 * B_s - 1 + defer_s = max(2 * B_s - 1, D_s)`, so `max(2 * B_s, D_s + 1)` samples
/// guarantees exact preservation of both initial zero padding (at `k = 0`) and prior input
/// history (for all `k >= 1`).
pub(crate) fn shared_input_ring_len(plan: &NonUniformPlan) -> Result<usize, ConvError> {
    let mut len = 0usize;
    for st in plan.stages() {
        if st.kind == NonUniformStageKind::Fdl {
            let two_b = st
                .block_size
                .checked_mul(2)
                .ok_or(ConvError::ArithmeticOverflow)?;
            let offset_plus_one = st
                .ir_offset
                .checked_add(1)
                .ok_or(ConvError::ArithmeticOverflow)?;
            let req = two_b.max(offset_plus_one);
            if req > len {
                len = req;
            }
        }
    }
    Ok(len)
}

/// Computes the exact workspace requirements for a validated [`NonUniformPlan`] and `FftBackend`.
pub fn compute_f32_memory_requirements<F: FftBackend>(
    plan: &NonUniformPlan,
    fft: &F,
) -> Result<NonUniformMemoryRequirements, ConvError> {
    let mut state_f32 = shared_input_ring_len(plan)?;
    let mut spectral_f32 = 0usize;
    let mut delay_f32 = 0usize;
    let mut max_fft_size = 0usize;
    let mut max_fft_scratch = 0usize;

    for st in plan.stages() {
        match st.kind {
            NonUniformStageKind::Direct => {
                // Direct FIR: ir_coeffs (ir_len) + circular history (ir_len)
                let needed = st
                    .ir_len
                    .checked_mul(2)
                    .ok_or(ConvError::ArithmeticOverflow)?;
                state_f32 = state_f32
                    .checked_add(needed)
                    .ok_or(ConvError::ArithmeticOverflow)?;
            }
            NonUniformStageKind::Fdl => {
                let fft_enum = FftSize::from_len(st.fft_size)?;
                if !fft.supports_size(fft_enum) {
                    return Err(ConvError::UnsupportedFftSize {
                        fft_size: st.fft_size,
                    });
                }
                let bins = st.block_size + 1;
                // Spectral:
                // - ir_re + ir_im: 2 * P * bins
                // - fdl_re + fdl_im: 2 * P * bins (newest slot also holds in-place staged RFFT)
                // - acc_re + acc_im: 2 * bins (also holds in-place staged IFFT)
                let part_bins = st
                    .num_partitions
                    .checked_mul(bins)
                    .ok_or(ConvError::ArithmeticOverflow)?;
                let stage_spec = part_bins
                    .checked_mul(4)
                    .and_then(|v| v.checked_add(bins.checked_mul(2)?))
                    .ok_or(ConvError::ArithmeticOverflow)?;
                spectral_f32 = spectral_f32
                    .checked_add(stage_spec)
                    .ok_or(ConvError::ArithmeticOverflow)?;

                // Power-of-two output ring: 2 * block_size
                let dlen = stage_delay_ring_len(st.block_size)?;
                delay_f32 = delay_f32
                    .checked_add(dlen)
                    .ok_or(ConvError::ArithmeticOverflow)?;

                if st.fft_size > max_fft_size {
                    max_fft_size = st.fft_size;
                }
                let sc = fft.scratch_len_f32(fft_enum);
                if sc > max_fft_scratch {
                    max_fft_scratch = sc;
                }
            }
        }
    }

    // Shared scratch: time_scratch (max_fft_size) + fft_scratch (max_fft_scratch)
    let scratch_f32 = max_fft_size
        .checked_add(max_fft_scratch)
        .ok_or(ConvError::ArithmeticOverflow)?;

    let total_f32 = state_f32
        .checked_add(spectral_f32)
        .and_then(|v| v.checked_add(delay_f32))
        .and_then(|v| v.checked_add(scratch_f32))
        .ok_or(ConvError::ArithmeticOverflow)?;
    let total_bytes = total_f32
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(ConvError::ArithmeticOverflow)?;

    Ok(NonUniformMemoryRequirements {
        state_f32,
        spectral_f32,
        delay_f32,
        scratch_f32,
        total_f32,
        total_bytes,
    })
}
