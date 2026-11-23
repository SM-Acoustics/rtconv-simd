//! Stage storage offsets and `f32` numeric execution kernels for [`crate::NonUniformConvolver`].

use crate::nonuniform::memory::stage_delay_ring_len;
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
    pub overlap_in_off: usize,
    pub ir_re_off: usize,
    pub ir_im_off: usize,
    pub fdl_re_off: usize,
    pub fdl_im_off: usize,
    pub acc_re_off: usize,
    pub acc_im_off: usize,
    pub delay_ring_off: usize,
    pub delay_ring_len: usize,
    pub fdl_head: usize,
    pub block_size: usize,
    pub fft_size: usize,
    pub bins: usize,
    pub num_partitions: usize,
    pub active_partitions: usize,
}

/// Complete workspace layout for all stages plus shared FFT scratch slices.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorkspaceLayout {
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
        let mut cursor = 0usize;
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
                    let dlen = stage_delay_ring_len(st.ir_offset, st.block_size)?;

                    let overlap_in_off = cursor;
                    cursor = cursor
                        .checked_add(st.fft_size)
                        .ok_or(ConvError::ArithmeticOverflow)?;
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
                        overlap_in_off,
                        ir_re_off,
                        ir_im_off,
                        fdl_re_off,
                        fdl_im_off,
                        acc_re_off,
                        acc_im_off,
                        delay_ring_off,
                        delay_ring_len: dlen,
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
    let len = st.direct_len;
    let head = st.direct_head;
    ws[st.direct_hist_off + head] = x;

    let (coeffs_and_before, hist_and_after) = ws.split_at_mut(st.direct_hist_off);
    let coeffs = &coeffs_and_before[st.direct_coeffs_off..st.direct_coeffs_off + active];
    let hist = &hist_and_after[..len];

    // Circular dot product: history[head], history[head-1], ..., wrap around.
    let first_seg = (head + 1).min(active);
    let mut acc = 0.0f32;
    // Reverse order of hist[head + 1 - first_seg ..= head] matches coeffs[0..first_seg].
    for k in 0..first_seg {
        acc += coeffs[k] * hist[head - k];
    }
    let rem = active - first_seg;
    if rem > 0 {
        let mut h_idx = len - 1;
        for c in &coeffs[first_seg..active] {
            acc += *c * hist[h_idx];
            h_idx = h_idx.saturating_sub(1);
        }
    }
    let _ = backend;

    st.direct_head = if head + 1 == len { 0 } else { head + 1 };
    acc
}

/// Runs RFFT on the completed `block_size` input window, shifts overlap-save history,
/// advances `fdl_head`, zeroes the stage accumulator, and runs CMAC for `part_start..part_end`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fdl_stage_start_block<F: FftBackend>(
    ws: &mut [f32],
    st: &mut StageLayout,
    time_scratch_off: usize,
    fft_scratch_off: usize,
    max_fft_scratch: usize,
    part_start: usize,
    part_end: usize,
    finish_ifft: bool,
    delay_write_pos: &mut usize,
    backend: SimdBackend,
    fft: &F,
) -> Result<(), ConvError> {
    let b = st.block_size;
    let n = st.fft_size;
    let bins = st.bins;
    let p_total = st.num_partitions;
    let fft_enum = FftSize::from_len(n)?;

    if st.active_partitions == 0 {
        // Shift overlap-save input and commit zeros into delay ring when finishing.
        ws.copy_within(
            st.overlap_in_off + b..st.overlap_in_off + n,
            st.overlap_in_off,
        );
        if finish_ifft {
            for k in 0..b {
                let idx = (*delay_write_pos + k) % st.delay_ring_len;
                ws[st.delay_ring_off + idx] = 0.0;
            }
            *delay_write_pos += b;
        }
        return Ok(());
    }

    // Advance FDL ring head and copy overlap_in[0..N] into time_scratch[0..N].
    st.fdl_head = if st.fdl_head + 1 == p_total {
        0
    } else {
        st.fdl_head + 1
    };
    ws.copy_within(st.overlap_in_off..st.overlap_in_off + n, time_scratch_off);
    // Immediately shift overlap-save second half into first half for upcoming samples.
    ws.copy_within(
        st.overlap_in_off + b..st.overlap_in_off + n,
        st.overlap_in_off,
    );

    // Run RFFT into fdl_re[fdl_head], fdl_im[fdl_head].
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

    fft.rfft_forward(fft_enum, time_in, out_re, out_im, sc)?;

    // Zero stage spectral accumulators.
    ws[st.acc_re_off..st.acc_re_off + bins].fill(0.0);
    ws[st.acc_im_off..st.acc_im_off + bins].fill(0.0);

    fdl_stage_cmac_and_maybe_ifft(
        ws,
        st,
        time_scratch_off,
        fft_scratch_off,
        max_fft_scratch,
        part_start,
        part_end,
        finish_ifft,
        delay_write_pos,
        backend,
        fft,
    )
}

/// Accumulates CMAC partitions `part_start..part_end` and optionally finishes IFFT + delay commit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fdl_stage_cmac_and_maybe_ifft<F: FftBackend>(
    ws: &mut [f32],
    st: &StageLayout,
    time_scratch_off: usize,
    fft_scratch_off: usize,
    max_fft_scratch: usize,
    part_start: usize,
    part_end: usize,
    finish_ifft: bool,
    delay_write_pos: &mut usize,
    backend: SimdBackend,
    fft: &F,
) -> Result<(), ConvError> {
    let b = st.block_size;
    let n = st.fft_size;
    let bins = st.bins;
    let p_total = st.num_partitions;
    let p_end = part_end.min(st.active_partitions);

    if part_start < p_end {
        // Disjoint slices: ir_re < ir_im < fdl_re < fdl_im < acc_re < acc_im.
        let (s_before_acc, s_from_acc) = ws.split_at_mut(st.acc_re_off);
        let (acc_re_region, acc_im_region) = s_from_acc.split_at_mut(st.acc_im_off - st.acc_re_off);
        let acc_re = &mut acc_re_region[..bins];
        let acc_im = &mut acc_im_region[..bins];

        for p in part_start..p_end {
            let slot = (st.fdl_head + p_total - p) % p_total;
            let x_re =
                &s_before_acc[st.fdl_re_off + slot * bins..st.fdl_re_off + (slot + 1) * bins];
            let x_im =
                &s_before_acc[st.fdl_im_off + slot * bins..st.fdl_im_off + (slot + 1) * bins];
            let h_re = &s_before_acc[st.ir_re_off + p * bins..st.ir_re_off + (p + 1) * bins];
            let h_im = &s_before_acc[st.ir_im_off + p * bins..st.ir_im_off + (p + 1) * bins];
            if backend.is_scalar() {
                simd_kernels::cmac_scalar(acc_re, acc_im, x_re, x_im, h_re, h_im);
            } else {
                simd_kernels::cmac_simd(backend.level(), acc_re, acc_im, x_re, x_im, h_re, h_im);
            }
        }
    }

    if finish_ifft {
        let fft_enum = FftSize::from_len(n)?;
        if st.active_partitions == 0 {
            for k in 0..b {
                let idx = (*delay_write_pos + k) % st.delay_ring_len;
                ws[st.delay_ring_off + idx] = 0.0;
            }
            *delay_write_pos += b;
            return Ok(());
        }
        // Disjoint borrows: acc_re < acc_im < time_scratch <= fft_scratch.
        let (before_time, from_time) = ws.split_at_mut(time_scratch_off);
        let (time_slice, fft_sc_slice) = from_time.split_at_mut(fft_scratch_off - time_scratch_off);
        let acc_re = &before_time[st.acc_re_off..st.acc_re_off + bins];
        let acc_im = &before_time[st.acc_im_off..st.acc_im_off + bins];
        let time_out = &mut time_slice[..n];
        let sc = &mut fft_sc_slice[..max_fft_scratch];

        fft.rfft_inverse(fft_enum, acc_re, acc_im, time_out, sc)?;

        // Commit valid 50% overlap-save second half (`time_out[b..2*b]`) into delay ring.
        for k in 0..b {
            let sample = ws[time_scratch_off + b + k];
            let idx = (*delay_write_pos + k) % st.delay_ring_len;
            ws[st.delay_ring_off + idx] = sample;
        }
        *delay_write_pos += b;
    }

    Ok(())
}
