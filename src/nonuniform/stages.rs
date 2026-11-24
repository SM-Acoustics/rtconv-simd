//! Stage storage offsets and `f32` numeric execution kernels for [`crate::NonUniformConvolver`].

use crate::nonuniform::memory::{shared_input_ring_len, stage_delay_ring_len};
use crate::nonuniform::plan::{MAX_NONUNIFORM_STAGES, NonUniformPlan, NonUniformStageKind};
use crate::simd_kernels;
use crate::{ConvError, FftBackend, FftSize, SimdBackend};

/// Pre-computed slice offsets into the caller's flat `&mut [f32]` workspace for one stage.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StageLayout {
    pub kind: u8, // 0 = Direct, 1 = Fdl
    // Direct FIR slices
    pub direct_coeffs_off: usize,
    pub direct_hist_off: usize,
    pub direct_len: usize,
    pub direct_active_len: usize,
    pub direct_head: usize,
    // FDL slices
    pub ir_re_off: usize,
    pub ir_im_off: usize,
    pub fdl_re_off: usize,
    pub fdl_im_off: usize,
    pub acc_re_off: usize,
    pub acc_im_off: usize,
    pub delay_ring_off: usize,
    pub delay_ring_len: usize,
    pub delay_ring_mask: usize,
    pub defer_samples: usize,
    pub rfft_steps: usize,
    pub ifft_steps: usize,
    pub fdl_head: usize,
    pub block_size: usize,
    pub fft_size: usize,
    pub bins: usize,
    pub num_partitions: usize,
    pub active_partitions: usize,
}

/// Complete workspace layout for shared input ring, all stages, and shared FFT scratch slices.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorkspaceLayout {
    pub shared_in_off: usize,
    pub shared_in_len: usize,
    pub shared_in_write_pos: usize,
    pub stages: [StageLayout; MAX_NONUNIFORM_STAGES],
    pub stage_count: usize,
    pub time_scratch_off: usize,
    pub fft_scratch_off: usize,
    pub max_fft_scratch: usize,
    pub total_f32: usize,
}

impl WorkspaceLayout {
    pub fn build<F: FftBackend>(
        plan: &NonUniformPlan,
        active_ir_len: usize,
        fft: &F,
    ) -> Result<Self, ConvError> {
        let mut stages = [StageLayout::default(); MAX_NONUNIFORM_STAGES];
        let shared_in_off = 0usize;
        let shared_in_len = shared_input_ring_len(plan)?;
        let mut cursor = shared_in_len;
        let mut max_fft_size = 0usize;
        let mut max_fft_scratch = 0usize;

        for (idx, st) in plan.stages().iter().enumerate() {
            match st.kind {
                NonUniformStageKind::Direct => {
                    let coeffs_off = cursor;
                    cursor = cursor
                        .checked_add(st.ir_len)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let hist_off = cursor;
                    cursor = cursor
                        .checked_add(st.ir_len)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let stage_active = active_ir_len.saturating_sub(st.ir_offset).min(st.ir_len);
                    stages[idx] = StageLayout {
                        kind: 0,
                        direct_coeffs_off: coeffs_off,
                        direct_hist_off: hist_off,
                        direct_len: st.ir_len,
                        direct_active_len: stage_active,
                        direct_head: 0,
                        ..StageLayout::default()
                    };
                }
                NonUniformStageKind::Fdl => {
                    let fft_enum = FftSize::from_len(st.fft_size)?;
                    if !fft.supports_size(fft_enum) {
                        return Err(ConvError::UnsupportedFftSize {
                            fft_size: st.fft_size,
                        });
                    }
                    let bins = st.block_size + 1;
                    let part_bins = st
                        .num_partitions
                        .checked_mul(bins)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let dlen = stage_delay_ring_len(st.block_size)?;
                    let defer_samples = st
                        .slack_samples
                        .saturating_sub(st.block_size.saturating_sub(1));
                    let rfft_steps = fft.staged_forward_steps(fft_enum).max(1);
                    let ifft_steps = fft.staged_inverse_steps(fft_enum).max(1);

                    let ir_re_off = cursor;
                    cursor = cursor
                        .checked_add(part_bins)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let ir_im_off = cursor;
                    cursor = cursor
                        .checked_add(part_bins)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let fdl_re_off = cursor;
                    cursor = cursor
                        .checked_add(part_bins)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let fdl_im_off = cursor;
                    cursor = cursor
                        .checked_add(part_bins)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let acc_re_off = cursor;
                    cursor = cursor
                        .checked_add(bins)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let acc_im_off = cursor;
                    cursor = cursor
                        .checked_add(bins)
                        .ok_or(ConvError::ArithmeticOverflow)?;
                    let delay_ring_off = cursor;
                    cursor = cursor
                        .checked_add(dlen)
                        .ok_or(ConvError::ArithmeticOverflow)?;

                    let stage_active = active_ir_len.saturating_sub(st.ir_offset).min(st.ir_len);
                    let active_partitions = if stage_active == 0 {
                        0
                    } else {
                        stage_active.div_ceil(st.block_size)
                    };

                    if st.fft_size > max_fft_size {
                        max_fft_size = st.fft_size;
                    }
                    let sc = fft.scratch_len_f32(fft_enum);
                    if sc > max_fft_scratch {
                        max_fft_scratch = sc;
                    }

                    stages[idx] = StageLayout {
                        kind: 1,
                        ir_re_off,
                        ir_im_off,
                        fdl_re_off,
                        fdl_im_off,
                        acc_re_off,
                        acc_im_off,
                        delay_ring_off,
                        delay_ring_len: dlen,
                        delay_ring_mask: dlen - 1,
                        defer_samples,
                        rfft_steps,
                        ifft_steps,
                        fdl_head: 0,
                        block_size: st.block_size,
                        fft_size: st.fft_size,
                        bins,
                        num_partitions: st.num_partitions,
                        active_partitions,
                        ..StageLayout::default()
                    };
                }
            }
        }

        let time_scratch_off = cursor;
        cursor = cursor
            .checked_add(max_fft_size)
            .ok_or(ConvError::ArithmeticOverflow)?;
        let fft_scratch_off = cursor;
        cursor = cursor
            .checked_add(max_fft_scratch)
            .ok_or(ConvError::ArithmeticOverflow)?;

        Ok(Self {
            shared_in_off,
            shared_in_len,
            shared_in_write_pos: 0,
            stages,
            stage_count: plan.stage_count(),
            time_scratch_off,
            fft_scratch_off,
            max_fft_scratch,
            total_f32: cursor,
        })
    }
}

/// Processes one sample through a Direct FIR stage using the selected `SimdBackend`.
///
/// Uses the same reverse-indexed circular history as [`crate::DirectConvolver`]: the newest
/// sample is written at a decrementing cursor so that both coefficient and history segments are
/// traversed forward with unit stride, which lets the dot product use the `fearless_simd`
/// kernels (`dot_two_segments_simd`) instead of a scalar reversed loop.
#[inline]
pub(crate) fn direct_stage_process_sample(
    ws: &mut [f32],
    st: &mut StageLayout,
    x: f32,
    backend: SimdBackend,
) -> f32 {
    let active = st.direct_active_len;
    if active == 0 {
        return 0.0;
    }
    let pos = if st.direct_head == 0 {
        active - 1
    } else {
        st.direct_head - 1
    };
    st.direct_head = pos;
    ws[st.direct_hist_off + pos] = x;

    let coeffs = &ws[st.direct_coeffs_off..st.direct_coeffs_off + active];
    let hist = &ws[st.direct_hist_off..st.direct_hist_off + active];
    let first_len = active - pos;
    if backend.is_scalar() {
        simd_kernels::dot_product_scalar(&coeffs[..first_len], &hist[pos..])
            + simd_kernels::dot_product_scalar(&coeffs[first_len..], &hist[..pos])
    } else {
        simd_kernels::dot_two_segments_simd(
            backend.level(),
            &coeffs[..first_len],
            &hist[pos..],
            &coeffs[first_len..],
            &hist[..pos],
        )
    }
}

/// Copies the `n = 2 * B_s` overlap-save input window ending `defer_samples` before
/// `in_write_pos` from the shared real input ring into `time_scratch[0..n]` (zero division).
#[inline]
fn copy_shared_input_window(
    ws: &mut [f32],
    shared_in_off: usize,
    shared_in_len: usize,
    in_write_pos: usize,
    defer_samples: usize,
    n: usize,
    time_scratch_off: usize,
) {
    let window_age = (n - 1) + defer_samples;
    let start = if in_write_pos >= window_age {
        in_write_pos - window_age
    } else {
        in_write_pos + shared_in_len - window_age
    };
    let first = (shared_in_len - start).min(n);
    ws.copy_within(
        shared_in_off + start..shared_in_off + start + first,
        time_scratch_off,
    );
    if first < n {
        ws.copy_within(
            shared_in_off..shared_in_off + (n - first),
            time_scratch_off + first,
        );
    }
}

/// Executes forward RFFT steps `step_start..step_end` in-place inside `fdl_re/im[fdl_head]`.
///
/// On `step_start == 0`, advances `fdl_head` and captures the `2 * B_s` overlap-save window
/// from the shared real input ring. On `finish_rfft`, flushes subnormals on the finished FDL
/// spectrum and zeroes the stage spectral accumulators for CMAC.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fdl_stage_run_rfft_steps<F: FftBackend>(
    ws: &mut [f32],
    st: &mut StageLayout,
    shared_in_off: usize,
    shared_in_len: usize,
    in_write_pos: usize,
    time_scratch_off: usize,
    fft_scratch_off: usize,
    max_fft_scratch: usize,
    step_start: usize,
    step_end: usize,
    finish_rfft: bool,
    fft: &F,
) -> Result<(), ConvError> {
    if st.active_partitions == 0 || step_start >= step_end {
        return Ok(());
    }

    let n = st.fft_size;
    let bins = st.bins;
    let p_total = st.num_partitions;
    let fft_enum = FftSize::from_len(n)?;

    if step_start == 0 {
        st.fdl_head = if st.fdl_head + 1 == p_total {
            0
        } else {
            st.fdl_head + 1
        };
        copy_shared_input_window(
            ws,
            shared_in_off,
            shared_in_len,
            in_write_pos,
            st.defer_samples,
            n,
            time_scratch_off,
        );
    }

    let slot_re_off = st.fdl_re_off + st.fdl_head * bins;
    let slot_im_off = st.fdl_im_off + st.fdl_head * bins;

    // Disjoint mutable borrows: fdl_re_off < fdl_im_off < time_scratch_off <= fft_scratch_off.
    let (before_im, from_im) = ws.split_at_mut(st.fdl_im_off);
    let (from_im_part, from_time) = from_im.split_at_mut(time_scratch_off - st.fdl_im_off);
    let (time_slice, fft_sc_slice) = from_time.split_at_mut(fft_scratch_off - time_scratch_off);

    let out_re = &mut before_im[slot_re_off..slot_re_off + bins];
    let out_im = &mut from_im_part[slot_im_off - st.fdl_im_off..slot_im_off - st.fdl_im_off + bins];
    let time_in = &time_slice[..n];
    let sc = &mut fft_sc_slice[..max_fft_scratch];

    if step_start == 0 && finish_rfft {
        fft.rfft_forward(fft_enum, time_in, out_re, out_im, sc)?;
    } else {
        for step in step_start..step_end {
            fft.rfft_forward_step(fft_enum, step, time_in, out_re, out_im, sc)?;
        }
    }

    if finish_rfft {
        // Cancellation can leave subnormal bins; they would be re-multiplied for P blocks.
        simd_kernels::flush_tiny_in_place(out_re);
        simd_kernels::flush_tiny_in_place(out_im);
        ws[st.acc_re_off..st.acc_re_off + bins].fill(0.0);
        ws[st.acc_im_off..st.acc_im_off + bins].fill(0.0);
    }

    Ok(())
}

/// Accumulates CMAC partitions `part_start..part_end` into `acc_re/im`.
#[inline]
pub(crate) fn fdl_stage_run_cmac(
    ws: &mut [f32],
    st: &StageLayout,
    part_start: usize,
    part_end: usize,
    backend: SimdBackend,
) {
    let bins = st.bins;
    let p_total = st.num_partitions;
    let p_end = part_end.min(st.active_partitions);

    if part_start >= p_end {
        return;
    }

    // Disjoint slices: ir_re < ir_im < fdl_re < fdl_im < acc_re < acc_im.
    let (s_before_acc, s_from_acc) = ws.split_at_mut(st.acc_re_off);
    let (acc_re_region, acc_im_region) = s_from_acc.split_at_mut(st.acc_im_off - st.acc_re_off);
    let acc_re = &mut acc_re_region[..bins];
    let acc_im = &mut acc_im_region[..bins];

    for p in part_start..p_end {
        let slot = if st.fdl_head >= p {
            st.fdl_head - p
        } else {
            st.fdl_head + p_total - p
        };
        let x_re = &s_before_acc[st.fdl_re_off + slot * bins..st.fdl_re_off + (slot + 1) * bins];
        let x_im = &s_before_acc[st.fdl_im_off + slot * bins..st.fdl_im_off + (slot + 1) * bins];
        let h_re = &s_before_acc[st.ir_re_off + p * bins..st.ir_re_off + (p + 1) * bins];
        let h_im = &s_before_acc[st.ir_im_off + p * bins..st.ir_im_off + (p + 1) * bins];
        if backend.is_scalar() {
            simd_kernels::cmac_scalar(acc_re, acc_im, x_re, x_im, h_re, h_im);
        } else {
            simd_kernels::cmac_simd(backend.level(), acc_re, acc_im, x_re, x_im, h_re, h_im);
        }
    }
}

/// Executes inverse IFFT steps `step_start..step_end` in-place inside `acc_re/im` and, when
/// `commit_output` is `true`, commits the valid OLS second half `[B_s..2*B_s]` into the
/// power-of-two `2 * B_s` output delay ring using branchless bitmask indexing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fdl_stage_run_ifft_steps<F: FftBackend>(
    ws: &mut [f32],
    st: &StageLayout,
    time_scratch_off: usize,
    fft_scratch_off: usize,
    max_fft_scratch: usize,
    step_start: usize,
    step_end: usize,
    commit_output: bool,
    delay_write_pos: &mut usize,
    fft: &F,
) -> Result<(), ConvError> {
    if step_start >= step_end {
        return Ok(());
    }

    let b = st.block_size;
    let n = st.fft_size;
    let bins = st.bins;
    let mask = st.delay_ring_mask;

    if st.active_partitions == 0 {
        if commit_output {
            for k in 0..b {
                let idx = (*delay_write_pos + k) & mask;
                ws[st.delay_ring_off + idx] = 0.0;
            }
            *delay_write_pos += b;
        }
        return Ok(());
    }

    let fft_enum = FftSize::from_len(n)?;
    if step_start == 0 {
        simd_kernels::flush_tiny_in_place(&mut ws[st.acc_re_off..st.acc_re_off + bins]);
        simd_kernels::flush_tiny_in_place(&mut ws[st.acc_im_off..st.acc_im_off + bins]);
    }

    // Disjoint mutable borrows: acc_re < acc_im < time_scratch <= fft_scratch.
    let (before_im, from_im) = ws.split_at_mut(st.acc_im_off);
    let (im_region, from_time) = from_im.split_at_mut(time_scratch_off - st.acc_im_off);
    let (time_slice, fft_sc_slice) = from_time.split_at_mut(fft_scratch_off - time_scratch_off);
    let acc_re = &mut before_im[st.acc_re_off..st.acc_re_off + bins];
    let acc_im = &mut im_region[..bins];
    let time_out = &mut time_slice[..n];
    let sc = &mut fft_sc_slice[..max_fft_scratch];

    if step_start == 0 && commit_output {
        fft.rfft_inverse(fft_enum, acc_re, acc_im, time_out, sc)?;
    } else {
        for step in step_start..step_end {
            fft.rfft_inverse_step(fft_enum, step, acc_re, acc_im, time_out, sc)?;
        }
    }

    if commit_output {
        for k in 0..b {
            let sample = ws[time_scratch_off + b + k];
            let idx = (*delay_write_pos + k) & mask;
            ws[st.delay_ring_off + idx] = sample;
        }
        *delay_write_pos += b;
    }

    Ok(())
}
