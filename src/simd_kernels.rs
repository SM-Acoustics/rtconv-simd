//! Scalar and `fearless_simd`-vectorized DSP kernels (100% safe, `no_std`).
//!
//! Kernels provided:
//! - Real dot product (FIR inner loop) — scalar reference and SIMD (`S::f32s` + FMA `mul_add`).
//! - Block-dispatched direct FIR loop — dispatches `Level` once per audio block rather than per sample.
//! - Split-complex (SoA) Complex Multiply-Accumulate (CMAC) — scalar reference and SIMD (`mul_add`/`mul_sub`).
//! - Block-dispatched FDL accumulation — dispatches `Level` once per partitioned FFT block.

use fearless_simd::{Level, Simd, dispatch, prelude::*};
use fearless_simd_macros::simd;

/// Magnitude below which `f32` input samples and filter coefficients are flushed to zero:
/// `2^-63` (≈ `1.08e-19`, ≈ −379 dBFS).
///
/// IEEE 754 subnormal operands and results are executed by microcode assists on x86 when the
/// host thread has not enabled FTZ/DAZ (Rust never does, and `#![forbid(unsafe_code)]` rules out
/// touching `MXCSR`), and are slow on several other FPUs. Measured on x86-64 without FTZ/DAZ,
/// a 1024-tap `DirectConvolver` ran 30–75× slower for inputs of `1e-35`..`1e-37` (normal numbers
/// whose products with the taps are subnormal) and 15× slower, permanently, with an IR whose deep
/// tail decoded to subnormal `f32` taps. Bounding every non-zero input **and** coefficient
/// magnitude by `2^-63` guarantees that every non-zero product is at least `2^-126`, i.e. a normal
/// number. The flushed values are ~380 dB below full scale, far under the 24-bit noise floor
/// (−144 dBFS) and under `f32` resolution of any audible signal.
pub const DENORMAL_GUARD: f32 = f32::from_bits(0x2000_0000);

/// Returns `0.0` when `|x| < DENORMAL_GUARD`, otherwise `x` (`NaN` and infinities pass through).
#[inline(always)]
pub fn flush_tiny(x: f32) -> f32 {
    if x.abs() < DENORMAL_GUARD { 0.0 } else { x }
}

/// Copies `src` into `dst` (up to the shorter length) applying [`flush_tiny`].
#[inline]
pub fn copy_flush_tiny(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = flush_tiny(s);
    }
}

/// Applies [`flush_tiny`] in place (used on coefficients and pre-transformed spectra).
#[inline]
pub fn flush_tiny_in_place(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = flush_tiny(*v);
    }
}

/// Scalar reference dot product over two equal-length slices.
///
/// Always executes scalar arithmetic without SIMD vectorization.
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    let mut i = 0;
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

/// SIMD-generic dot product over two equal-length slices using native `S::f32s` width.
///
/// Uses dual vector accumulators to hide FMA latency and handles arbitrary
/// non-multiple tail lengths safely.
#[inline(always)]
fn dot_product_vec<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let lanes = S::f32s::LEN;
    let two_lanes = lanes * 2;

    let mut acc0 = S::f32s::splat(simd, 0.0);
    let mut acc1 = S::f32s::splat(simd, 0.0);
    let mut i = 0;

    while i + two_lanes <= n {
        let a0 = S::f32s::from_slice(simd, &a[i..i + lanes]);
        let b0 = S::f32s::from_slice(simd, &b[i..i + lanes]);
        acc0 = a0.mul_add(b0, acc0);

        let a1 = S::f32s::from_slice(simd, &a[i + lanes..i + two_lanes]);
        let b1 = S::f32s::from_slice(simd, &b[i + lanes..i + two_lanes]);
        acc1 = a1.mul_add(b1, acc1);

        i += two_lanes;
    }

    while i + lanes <= n {
        let av = S::f32s::from_slice(simd, &a[i..i + lanes]);
        let bv = S::f32s::from_slice(simd, &b[i..i + lanes]);
        acc0 = av.mul_add(bv, acc0);
        i += lanes;
    }

    let combined = acc0 + acc1;
    let mut tmp = [0.0f32; 16];
    let store_len = lanes.min(tmp.len());
    combined.store_slice(&mut tmp[..store_len]);

    let mut sum = 0.0f32;
    let mut k = 0;
    while k < store_len {
        sum += tmp[k];
        k += 1;
    }

    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

#[simd]
fn dot_product_simd_impl<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    dot_product_vec(simd, a, b)
}

/// Dispatched SIMD dot product using a pre-validated `fearless_simd::Level`.
#[inline]
pub fn dot_product_simd(level: Level, a: &[f32], b: &[f32]) -> f32 {
    dispatch!(level, simd => dot_product_simd_impl(simd, a, b))
}

#[simd]
fn dot_two_segments_impl<S: Simd>(
    simd: S,
    ir_first: &[f32],
    hist_tail: &[f32],
    ir_second: &[f32],
    hist_head: &[f32],
) -> f32 {
    dot_product_vec(simd, ir_first, hist_tail) + dot_product_vec(simd, ir_second, hist_head)
}

/// Computes a single FIR sample across a circular buffer split in one `dispatch!` call.
#[inline]
pub fn dot_two_segments_simd(
    level: Level,
    ir_first: &[f32],
    hist_tail: &[f32],
    ir_second: &[f32],
    hist_head: &[f32],
) -> f32 {
    dispatch!(
        level,
        simd => dot_two_segments_impl(simd, ir_first, hist_tail, ir_second, hist_head)
    )
}

#[simd]
fn fir_block_impl<S: Simd>(
    simd: S,
    ir: &[f32],
    history: &mut [f32],
    write_pos: &mut usize,
    input: &[f32],
    output: &mut [f32],
) {
    let ir_len = ir.len();
    let n = input.len().min(output.len());
    if ir_len == 0 || history.len() < ir_len {
        return;
    }

    let mut pos = *write_pos;
    if pos >= ir_len {
        pos = 0;
    }

    let mut s = 0;
    while s < n {
        pos = if pos == 0 { ir_len - 1 } else { pos - 1 };
        history[pos] = flush_tiny(input[s]);

        let first_len = ir_len - pos;
        let y = dot_product_vec(simd, &ir[..first_len], &history[pos..ir_len])
            + dot_product_vec(simd, &ir[first_len..ir_len], &history[..pos]);
        output[s] = flush_tiny(y);
        s += 1;
    }

    *write_pos = pos;
}

/// Processes an entire FIR block with a single `dispatch!` call at block entry.
#[inline]
pub fn fir_block_simd(
    level: Level,
    ir: &[f32],
    history: &mut [f32],
    write_pos: &mut usize,
    input: &[f32],
    output: &mut [f32],
) {
    dispatch!(
        level,
        simd => fir_block_impl(simd, ir, history, write_pos, input, output)
    );
}

/// Scalar reference block FIR loop (for explicit comparison and scalar backend).
#[inline]
pub fn fir_block_scalar(
    ir: &[f32],
    history: &mut [f32],
    write_pos: &mut usize,
    input: &[f32],
    output: &mut [f32],
) {
    let ir_len = ir.len();
    let n = input.len().min(output.len());
    if ir_len == 0 || history.len() < ir_len {
        return;
    }

    let mut pos = *write_pos;
    if pos >= ir_len {
        pos = 0;
    }

    let mut s = 0;
    while s < n {
        pos = if pos == 0 { ir_len - 1 } else { pos - 1 };
        history[pos] = flush_tiny(input[s]);

        let first_len = ir_len - pos;
        let y = dot_product_scalar(&ir[..first_len], &history[pos..ir_len])
            + dot_product_scalar(&ir[first_len..ir_len], &history[..pos]);
        output[s] = flush_tiny(y);
        s += 1;
    }

    *write_pos = pos;
}

/// Scalar reference split-complex (SoA) complex multiply-accumulate:
///
/// `acc_re[k] += x_re[k] * h_re[k] - x_im[k] * h_im[k]`
/// `acc_im[k] += x_re[k] * h_im[k] + x_im[k] * h_re[k]`
#[inline]
pub fn cmac_scalar(
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
) {
    let n = acc_re
        .len()
        .min(acc_im.len())
        .min(x_re.len())
        .min(x_im.len())
        .min(h_re.len())
        .min(h_im.len());
    let mut k = 0;
    while k < n {
        let xr = x_re[k];
        let xi = x_im[k];
        let hr = h_re[k];
        let hi = h_im[k];
        acc_re[k] += xr * hr - xi * hi;
        acc_im[k] += xr * hi + xi * hr;
        k += 1;
    }
}

/// SIMD-generic split-complex (SoA) CMAC inner loop using `mul_add` / `mul_sub`.
#[inline(always)]
fn cmac_vec<S: Simd>(
    simd: S,
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
) {
    let n = acc_re
        .len()
        .min(acc_im.len())
        .min(x_re.len())
        .min(x_im.len())
        .min(h_re.len())
        .min(h_im.len());
    let lanes = S::f32s::LEN;
    let mut k = 0;

    while k + lanes <= n {
        let xr = S::f32s::from_slice(simd, &x_re[k..k + lanes]);
        let xi = S::f32s::from_slice(simd, &x_im[k..k + lanes]);
        let hr = S::f32s::from_slice(simd, &h_re[k..k + lanes]);
        let hi = S::f32s::from_slice(simd, &h_im[k..k + lanes]);

        let ar = S::f32s::from_slice(simd, &acc_re[k..k + lanes]);
        let ai = S::f32s::from_slice(simd, &acc_im[k..k + lanes]);

        // real: acc_re + (xr * hr - xi * hi) = (xr * hr + acc_re) - xi * hi
        let re_tmp = xr.mul_add(hr, ar);
        let new_re = re_tmp - xi * hi;

        // imag: acc_im + (xr * hi + xi * hr)
        let im_tmp = xr.mul_add(hi, ai);
        let new_im = xi.mul_add(hr, im_tmp);

        new_re.store_slice(&mut acc_re[k..k + lanes]);
        new_im.store_slice(&mut acc_im[k..k + lanes]);

        k += lanes;
    }

    // Tail handling (e.g., Nyquist bin at index N/2 when bins = N/2 + 1).
    while k < n {
        let xr = x_re[k];
        let xi = x_im[k];
        let hr = h_re[k];
        let hi = h_im[k];
        acc_re[k] += xr * hr - xi * hi;
        acc_im[k] += xr * hi + xi * hr;
        k += 1;
    }
}

#[simd]
fn cmac_simd_impl<S: Simd>(
    simd: S,
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
) {
    cmac_vec(simd, acc_re, acc_im, x_re, x_im, h_re, h_im);
}

/// Dispatched SIMD split-complex CMAC.
#[inline]
pub fn cmac_simd(
    level: Level,
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
) {
    dispatch!(
        level,
        simd => cmac_simd_impl(simd, acc_re, acc_im, x_re, x_im, h_re, h_im)
    );
}

#[allow(clippy::too_many_arguments)]
#[simd]
fn cmac_fdl_impl<S: Simd>(
    simd: S,
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    fdl_re: &[f32],
    fdl_im: &[f32],
    ir_re: &[f32],
    ir_im: &[f32],
    num_partitions: usize,
    bins: usize,
    fdl_head: usize,
) {
    if num_partitions == 0 || bins == 0 {
        return;
    }
    let mut p = 0;
    let mut ring_idx = fdl_head;
    while p < num_partitions {
        let x_off = ring_idx * bins;
        let h_off = p * bins;
        if x_off + bins <= fdl_re.len()
            && x_off + bins <= fdl_im.len()
            && h_off + bins <= ir_re.len()
            && h_off + bins <= ir_im.len()
        {
            cmac_vec(
                simd,
                acc_re,
                acc_im,
                &fdl_re[x_off..x_off + bins],
                &fdl_im[x_off..x_off + bins],
                &ir_re[h_off..h_off + bins],
                &ir_im[h_off..h_off + bins],
            );
        }
        ring_idx = if ring_idx == 0 {
            num_partitions - 1
        } else {
            ring_idx - 1
        };
        p += 1;
    }
}

/// Accumulates all FDL partitions in a single `dispatch!` call.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn cmac_fdl_simd(
    level: Level,
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    fdl_re: &[f32],
    fdl_im: &[f32],
    ir_re: &[f32],
    ir_im: &[f32],
    num_partitions: usize,
    bins: usize,
    fdl_head: usize,
) {
    dispatch!(
        level,
        simd => cmac_fdl_impl(
            simd,
            acc_re,
            acc_im,
            fdl_re,
            fdl_im,
            ir_re,
            ir_im,
            num_partitions,
            bins,
            fdl_head
        )
    );
}

/// Scalar reference FDL accumulation across all partitions.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn cmac_fdl_scalar(
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    fdl_re: &[f32],
    fdl_im: &[f32],
    ir_re: &[f32],
    ir_im: &[f32],
    num_partitions: usize,
    bins: usize,
    fdl_head: usize,
) {
    cmac_fdl_range_scalar(
        acc_re,
        acc_im,
        fdl_re,
        fdl_im,
        ir_re,
        ir_im,
        num_partitions,
        bins,
        fdl_head,
        0,
        num_partitions,
    );
}

#[simd]
#[allow(clippy::too_many_arguments)]
fn cmac_fdl_range_impl<S: Simd>(
    simd: S,
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    fdl_re: &[f32],
    fdl_im: &[f32],
    ir_re: &[f32],
    ir_im: &[f32],
    num_partitions: usize,
    bins: usize,
    ring_start_idx: usize,
    start_p: usize,
    end_p: usize,
) {
    if num_partitions == 0 || bins == 0 || start_p >= end_p {
        return;
    }
    let limit = end_p.min(num_partitions);
    let mut p = start_p;
    let mut ring_idx = ring_start_idx % num_partitions;
    while p < limit {
        let x_off = ring_idx * bins;
        let h_off = p * bins;
        if x_off + bins <= fdl_re.len()
            && x_off + bins <= fdl_im.len()
            && h_off + bins <= ir_re.len()
            && h_off + bins <= ir_im.len()
        {
            cmac_vec(
                simd,
                acc_re,
                acc_im,
                &fdl_re[x_off..x_off + bins],
                &fdl_im[x_off..x_off + bins],
                &ir_re[h_off..h_off + bins],
                &ir_im[h_off..h_off + bins],
            );
        }
        ring_idx = if ring_idx == 0 {
            num_partitions - 1
        } else {
            ring_idx - 1
        };
        p += 1;
    }
}

/// Accumulates a partition sub-range `[start_p, end_p)` in a single `dispatch!` call.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn cmac_fdl_range_simd(
    level: Level,
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    fdl_re: &[f32],
    fdl_im: &[f32],
    ir_re: &[f32],
    ir_im: &[f32],
    num_partitions: usize,
    bins: usize,
    ring_start_idx: usize,
    start_p: usize,
    end_p: usize,
) {
    dispatch!(
        level,
        simd => cmac_fdl_range_impl(
            simd,
            acc_re,
            acc_im,
            fdl_re,
            fdl_im,
            ir_re,
            ir_im,
            num_partitions,
            bins,
            ring_start_idx,
            start_p,
            end_p
        )
    );
}

/// Scalar reference FDL accumulation over a partition sub-range `[start_p, end_p)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn cmac_fdl_range_scalar(
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    fdl_re: &[f32],
    fdl_im: &[f32],
    ir_re: &[f32],
    ir_im: &[f32],
    num_partitions: usize,
    bins: usize,
    ring_start_idx: usize,
    start_p: usize,
    end_p: usize,
) {
    if num_partitions == 0 || bins == 0 || start_p >= end_p {
        return;
    }
    let limit = end_p.min(num_partitions);
    let mut p = start_p;
    let mut ring_idx = ring_start_idx % num_partitions;
    while p < limit {
        let x_off = ring_idx * bins;
        let h_off = p * bins;
        if x_off + bins <= fdl_re.len()
            && x_off + bins <= fdl_im.len()
            && h_off + bins <= ir_re.len()
            && h_off + bins <= ir_im.len()
        {
            cmac_scalar(
                acc_re,
                acc_im,
                &fdl_re[x_off..x_off + bins],
                &fdl_im[x_off..x_off + bins],
                &ir_re[h_off..h_off + bins],
                &ir_im[h_off..h_off + bins],
            );
        }
        ring_idx = if ring_idx == 0 {
            num_partitions - 1
        } else {
            ring_idx - 1
        };
        p += 1;
    }
}
