//! 32-bit-mantissa block-floating-point (BFP) real FFT core for `N = 64 ..= 2048`
//! (`fft_q31_bfp`), shared by the Q1.15 engines and usable by a Q1.31 engine family.
//!
//! Zero heap allocation, `#![forbid(unsafe_code)]`, integer arithmetic only.
//!
//! ## Arithmetic
//! - **Mantissas `i32`, twiddles Q1.31** from a single quarter-wave ROM
//!   (`TWIDDLE_COS_Q31_2048`, `513 × 4 = 2,052 B`). Each real product is `i32 × i32 → i64`
//!   (`mul` + `mulh` on RV32IM) and the complex product is rounded back to `i32` once, with
//!   round-to-nearest-even (`>> 31`). The unit twiddles `W^0 = 1` and `W^{N/4} = ∓j` are exact
//!   (`2^31` in `i64`; `W^0` skips the multiply).
//! - **Block exponent with per-stage headroom detection.** The OR of the magnitudes written by
//!   each pass is tracked; the next pass right-shifts its inputs (round-to-nearest-even, fused
//!   into the loads) **only if** they could overflow: a radix-2 stage needs `|v| <= 2^29`
//!   (`|a ± W·b| <= (1 + √2)·2^29 < 2^31`), the (un)tangling pass, whose outputs are kept
//!   doubled to avoid a halving rounding, needs `|v| <= 2^28` (`(2 + 2√2)·2^28 < 2^31`). Every
//!   shift is added to the block exponent. Inputs are normalized to `2^29` (forward) or `2^28`
//!   (inverse) first, so quiet blocks keep ~29 significant bits (Kalliojärvi & Astola 1996).
//!
//! ## Real-input packing (Sorensen, Jones, Heideman & Burrus, IEEE Trans. ASSP 35(6), 1987, §II)
//! The length-`N` real sequence is packed as `z(n) = x(2n) + j·x(2n+1)` (Eq. 6), transformed by
//! a length-`M = N/2` complex FFT, separated with Eq. (8) and recombined with Eq. (9):
//! `X(k) = E(k) + W_N^k·O(k)`, `E(k) = (Z(k) + Z*(M−k))/2`, `O(k) = (Z(k) − Z*(M−k))/(2j)`.
//! Only `X(0..=M)` is produced (Hermitian symmetry, Eqs. 2, 10, 11). The inverse reverses the
//! steps (`Z(k) = E(k) + j·O(k)`, `O(k) = (X(k) − X*(M−k))·W_N^{−k}/2`) and a length-`M` inverse
//! complex FFT. This halves the butterfly count of the former length-`N` complex transform of
//! real data (verified by the deterministic operation counter in the unit tests).
//!
//! Spectral layout and scale match `SimdFixedFftBackend`: forward is unnormalized
//! `X(k) = Σ x(n)·e^{−j2πnk/N}` for `k = 0..=N/2`, the inverse includes `1/N`.
//!
//! ## Buffers and time distribution
//! The transforms run in place inside a caller buffer of `i32` words, either a native `[i32]`
//! or `[i16]` with two words per value ([`Words32`]); forward needs [`q31_rfft_words`]`(N)`
//! words (`re[0..=M]` then `im[0..=M]`), inverse [`q31_irfft_words`]`(N)` (spectrum, then the
//! packed time sequence). Both are split into [`q31_fft_steps`]`(N) = log2(N)` bounded steps
//! for the non-uniform scheduler; running the steps back to back is the burst transform, so
//! staged and burst execution are bit-identical by construction.

use crate::fixed::block_float::{
    load_i32_from_i16, round_shift_even_i32, round_shift_even_i64, store_i32_to_i16,
};
use crate::fixed::fft_q31_tables::{MAX_Q31_FFT_SIZE, Q31_TW_QUARTER, TWIDDLE_COS_Q31_2048};

/// Smallest supported real FFT length.
pub(crate) const Q31_FFT_MIN: usize = 64;
/// Largest supported real FFT length (ROM-limited).
pub(crate) const Q31_FFT_MAX: usize = MAX_Q31_FFT_SIZE;

/// Input bound of a radix-2 stage (`|v| <= 2^29`).
const STAGE_LIMIT_BITS: u32 = 29;
/// Input bound of the doubled (un)tangling pass (`|v| <= 2^28`).
const PAIR_LIMIT_BITS: u32 = 28;

/// Block exponent marking an all-zero spectrum (never the maximum of a product exponent set).
pub(crate) const ZERO_SPECTRUM_EXP: i32 = i16::MIN as i32;

/// Storage of `i32` words: native `[i32]` or `[i16]` with two big-endian words per value.
pub(crate) trait Words32 {
    /// Loads word `i`.
    fn ld(&self, i: usize) -> i32;
    /// Stores word `i`.
    fn st(&mut self, i: usize, v: i32);
}

impl Words32 for [i32] {
    #[inline(always)]
    fn ld(&self, i: usize) -> i32 {
        self[i]
    }
    #[inline(always)]
    fn st(&mut self, i: usize, v: i32) {
        self[i] = v;
    }
}

impl Words32 for [i16] {
    #[inline(always)]
    fn ld(&self, i: usize) -> i32 {
        load_i32_from_i16(self, 2 * i)
    }
    #[inline(always)]
    fn st(&mut self, i: usize, v: i32) {
        store_i32_to_i16(self, 2 * i, v);
    }
}

/// `i64` accumulators stored in the same buffer (`[i16]`: four big-endian words, `[i32]`: two
/// big-endian words per value).
pub(crate) trait Words64: Words32 {
    /// Loads `i64` slot `i`.
    fn ld64(&self, i: usize) -> i64;
}

impl Words64 for [i16] {
    #[inline(always)]
    fn ld64(&self, i: usize) -> i64 {
        crate::fixed::block_float::load_i64_from_i16(self, 4 * i)
    }
}

impl Words64 for [i32] {
    #[inline(always)]
    fn ld64(&self, i: usize) -> i64 {
        ((self[2 * i] as i64) << 32) | (self[2 * i + 1] as u32 as i64)
    }
}

/// `true` if `n` is a supported real FFT length (power of two in `64..=2048`).
#[inline]
pub(crate) const fn q31_fft_supports(n: usize) -> bool {
    n.is_power_of_two() && n >= Q31_FFT_MIN && n <= Q31_FFT_MAX
}

/// Number of bounded steps of one forward or inverse transform (`log2(N)`).
#[inline]
pub(crate) const fn q31_fft_steps(n: usize) -> usize {
    n.trailing_zeros() as usize
}

/// `i32` words of the forward buffer: `re[0..=M]`, `im[0..=M]` (`N + 2`).
#[inline]
pub(crate) const fn q31_rfft_words(n: usize) -> usize {
    n + 2
}

/// `i32` words of the inverse buffer: spectrum (`N + 2`) followed by the packed time sequence
/// (`N`).
#[inline]
pub(crate) const fn q31_irfft_words(n: usize) -> usize {
    2 * n + 2
}

/// Per-transform state carried across steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Q31FftState {
    exp: i32,
    or_abs: u32,
    zero: bool,
}

impl Q31FftState {
    /// Fresh state.
    pub(crate) const fn new() -> Self {
        Self {
            exp: 0,
            or_abs: 0,
            zero: false,
        }
    }
}

#[inline(always)]
const fn bitlen(v: u32) -> u32 {
    32 - v.leading_zeros()
}

#[inline(always)]
const fn bitlen64(v: u64) -> u32 {
    64 - v.leading_zeros()
}

/// Right shift needed so that values whose OR of magnitudes is `or_abs` satisfy
/// `|v| <= 2^limit` after a round-to-nearest-even shift.
#[inline(always)]
const fn headroom_shift(or_abs: u32, limit: u32) -> u32 {
    bitlen(or_abs).saturating_sub(limit)
}

/// `RNE(v / 2^s)` (`s <= 30`), exact for `s = 0`; pure `i32` arithmetic (BFP shifts are
/// at most 4 bits).
#[inline(always)]
fn shr_rne(v: i32, s: u32) -> i32 {
    if s == 0 {
        v
    } else {
        round_shift_even_i32(v, s)
    }
}

/// Scales `v` by `2^sh` (`sh > 0`: exact left shift of a value known to fit; `sh < 0`: RNE).
#[inline(always)]
fn scale_to_fit(v: i32, sh: i32) -> i32 {
    if sh >= 0 {
        v << sh
    } else {
        shr_rne(v, sh.unsigned_abs())
    }
}

/// `RNE(p / 2^31)` for a complex-product sum `p` (`|p| < 2^62`); the bounds of the callers keep
/// the result inside `i32`.
#[inline(always)]
fn rne31(p: i64) -> i32 {
    round_shift_even_i64(p, 31) as i32
}

#[inline(always)]
fn bit_reverse(x: usize, bits: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - bits)
}

/// Deterministic operation counter for the unit tests (butterfly passes and complex products).
#[cfg(test)]
pub(crate) mod opcount {
    use core::cell::Cell;
    std::thread_local! {
        static PAIRS: Cell<u64> = const { Cell::new(0) };
        static CMULS: Cell<u64> = const { Cell::new(0) };
    }
    pub(crate) fn note(pairs: u64, cmuls: u64) {
        PAIRS.with(|c| c.set(c.get() + pairs));
        CMULS.with(|c| c.set(c.get() + cmuls));
    }
    /// Returns `(butterfly-equivalent pair operations, complex multiplications)` and resets.
    pub(crate) fn take() -> (u64, u64) {
        (PAIRS.with(|c| c.replace(0)), CMULS.with(|c| c.replace(0)))
    }
}

#[inline(always)]
fn count(_pairs: u64, _cmuls: u64) {
    #[cfg(test)]
    opcount::note(_pairs, _cmuls);
}

/// Twiddle `(cos, sin)` of `2π·idx/2048` as Q1.31 `i32` for `0 < idx < 1024`, `idx != 512`
/// (the unit twiddles are applied exactly by the callers).
#[inline(always)]
fn twiddle_i32(idx: usize) -> (i32, i32) {
    const Q: usize = Q31_TW_QUARTER;
    if idx < Q {
        (TWIDDLE_COS_Q31_2048[idx], TWIDDLE_COS_Q31_2048[Q - idx])
    } else {
        let rem = idx - Q;
        (-TWIDDLE_COS_Q31_2048[Q - rem], TWIDDLE_COS_Q31_2048[rem])
    }
}

/// `RNE((w·b − v·c) / 2^31)` with `i32 × i32 → i64` products (`mul` + `mulh` on RV32IM).
#[inline(always)]
fn cmul_part(w: i32, b: i32, v: i32, c: i32) -> i32 {
    rne31(w as i64 * b as i64 - v as i64 * c as i64)
}

/// One radix-2 decimation-in-time stage on the `m`-point complex sequence stored at
/// `re0..re0+m` / `im0..im0+m` (inputs in bit-reversed order before the first stage), with the
/// BFP pre-shift `s` fused into the loads. Returns the OR of the output magnitudes.
///
/// `W^0 = 1` and `W^{N/4} = ∓j` are applied exactly (no multiply); every other twiddle is a
/// Q1.31 `i32` and each output component is rounded once.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn cfft_stage<B: Words32 + ?Sized>(
    buf: &mut B,
    re0: usize,
    im0: usize,
    m: usize,
    half: usize,
    inverse: bool,
    s: u32,
) -> u32 {
    let step = half << 1;
    let tw_stride = MAX_Q31_FFT_SIZE / step;
    let quarter = half / 2;
    let mut or_abs = 0u32;
    for k in 0..half {
        let (wr, ws) = if k == 0 || k == quarter {
            (0, 0)
        } else {
            twiddle_i32(k * tw_stride)
        };
        let wi = if inverse { ws } else { -ws };
        let mut i0 = k;
        while i0 < m {
            let i1 = i0 + half;
            let ar = shr_rne(buf.ld(re0 + i0), s);
            let ai = shr_rne(buf.ld(im0 + i0), s);
            let br = shr_rne(buf.ld(re0 + i1), s);
            let bi = shr_rne(buf.ld(im0 + i1), s);
            let (tr, ti) = if k == 0 {
                (br, bi)
            } else if k == quarter {
                // W = -j (forward) or +j (inverse), exact.
                if inverse { (-bi, br) } else { (bi, -br) }
            } else {
                (cmul_part(wr, br, wi, bi), cmul_part(wr, bi, -wi, br))
            };
            let (r0, j0, r1, j1) = (ar + tr, ai + ti, ar - tr, ai - ti);
            buf.st(re0 + i0, r0);
            buf.st(im0 + i0, j0);
            buf.st(re0 + i1, r1);
            buf.st(im0 + i1, j1);
            or_abs |= r0.unsigned_abs() | j0.unsigned_abs() | r1.unsigned_abs() | j1.unsigned_abs();
            i0 += step;
        }
    }
    let groups = (m / step) as u64;
    let exact = if half >= 2 { 2 } else { 1 };
    count(half as u64 * groups, (half as u64 - exact) * groups);
    or_abs
}

/// One step of the staged forward real FFT (`step in 0..q31_fft_steps(n)`).
///
/// `x(i)` (`i in 0..n`) is the real input as `i32` with value `x(i)·2^e_in`; it is read only at
/// step `0`. After the last step, `buf` holds `X(k)` for `k = 0..=n/2` (`re` at word `k`, `im`
/// at word `n/2 + 1 + k`) and the function returns `Some(e)` with `X(k) = mant·2^e`
/// (`ZERO_SPECTRUM_EXP` for an all-zero input, whose mantissas are zero). Steps:
/// `0` = normalize + bit-reversed packing + first stage, `1..log2(n)-1` = stages,
/// `log2(n)-1` = untangling.
///
/// Inlined into its callers so the per-sample call chain of the engines carries one frame
/// less on RV32 (the stage and (un)tangling passes stay out of line).
#[inline(always)]
pub(crate) fn q31_rfft_step<B: Words32 + ?Sized, F: Fn(usize) -> i32>(
    n: usize,
    step: usize,
    x: F,
    e_in: i32,
    buf: &mut B,
    st: &mut Q31FftState,
) -> Option<i32> {
    debug_assert!(q31_fft_supports(n) && step < q31_fft_steps(n));
    let m = n / 2;
    let lm = m.trailing_zeros();
    let im0 = m + 1;
    let last = q31_fft_steps(n) - 1;

    if step == 0 {
        let mut or_abs = 0u32;
        for i in 0..n {
            or_abs |= x(i).unsigned_abs();
        }
        if or_abs == 0 {
            st.zero = true;
            st.exp = ZERO_SPECTRUM_EXP;
        } else {
            st.zero = false;
            // Normalize so the peak uses STAGE_LIMIT_BITS bits.
            let sh = STAGE_LIMIT_BITS as i32 - bitlen(or_abs) as i32;
            for k in 0..m {
                let src = bit_reverse(k, lm);
                buf.st(k, scale_to_fit(x(2 * src), sh));
                buf.st(im0 + k, scale_to_fit(x(2 * src + 1), sh));
            }
            st.exp = e_in - sh;
            st.or_abs = cfft_stage(buf, 0, im0, m, 1, false, 0);
        }
    } else if step < last {
        if !st.zero {
            let s = headroom_shift(st.or_abs, STAGE_LIMIT_BITS);
            st.exp += s as i32;
            st.or_abs = cfft_stage(buf, 0, im0, m, 1 << step, false, s);
        }
    } else if !st.zero {
        untangle(buf, m, st);
    }

    if step == last {
        if st.zero {
            for i in 0..q31_rfft_words(n) {
                buf.st(i, 0);
            }
        }
        Some(st.exp)
    } else {
        None
    }
}

/// Sorensen et al. (1987) Eq. (8)–(9) in place, with doubled outputs `2·X(k)` (exponent − 1)
/// so no halving is rounded.
#[inline(never)]
fn untangle<B: Words32 + ?Sized>(buf: &mut B, m: usize, st: &mut Q31FftState) {
    let im0 = m + 1;
    let s = headroom_shift(st.or_abs, PAIR_LIMIT_BITS);
    let mut or_abs = 0u32;

    // k = 0 (DC) and k = M (Nyquist): X(0) = Zr(0) + Zi(0), X(M) = Zr(0) − Zi(0).
    let a0 = shr_rne(buf.ld(0), s);
    let b0 = shr_rne(buf.ld(im0), s);
    let (dc, ny) = (2 * (a0 + b0), 2 * (a0 - b0));
    buf.st(0, dc);
    buf.st(im0, 0);
    buf.st(m, ny);
    buf.st(im0 + m, 0);
    or_abs |= dc.unsigned_abs() | ny.unsigned_abs();

    // k = M/2: W_N^{N/4} = −j, X(M/2) = conj(Z(M/2)).
    let h = m / 2;
    let hr = 2 * shr_rne(buf.ld(h), s);
    let hi = -2 * shr_rne(buf.ld(im0 + h), s);
    buf.st(h, hr);
    buf.st(im0 + h, hi);
    or_abs |= hr.unsigned_abs() | hi.unsigned_abs();

    let stride = MAX_Q31_FFT_SIZE / (2 * m);
    for k in 1..h {
        let mk = m - k;
        let a = shr_rne(buf.ld(k), s);
        let b = shr_rne(buf.ld(im0 + k), s);
        let c = shr_rne(buf.ld(mk), s);
        let d = shr_rne(buf.ld(im0 + mk), s);
        // 2E = (a + c) + j(b − d), 2O = (b + d) + j(c − a)
        let (er, ei) = (a + c, b - d);
        let (or, oi) = (b + d, c - a);
        let (wc, ws) = twiddle_i32(k * stride);
        // W_N^k · 2O with W_N^k = wc − j·ws
        let wbr = cmul_part(wc, or, -ws, oi);
        let wbi = cmul_part(wc, oi, ws, or);
        let (xr, xi, yr, yi) = (er + wbr, ei + wbi, er - wbr, -ei + wbi);
        buf.st(k, xr);
        buf.st(im0 + k, xi);
        buf.st(mk, yr);
        buf.st(im0 + mk, yi);
        or_abs |= xr.unsigned_abs() | xi.unsigned_abs() | yr.unsigned_abs() | yi.unsigned_abs();
    }
    count(h as u64, h.saturating_sub(1) as u64);
    st.exp += s as i32 - 1;
    st.or_abs = or_abs;
}

/// One step of the staged inverse real FFT (`step in 0..q31_fft_steps(n)`).
///
/// Before step `0`, `buf` holds `X(k)`, `k = 0..=n/2`, in the forward layout with value
/// `mant·2^e_x` (`e_x` is read at step `0` only); any magnitude below `2^31` is accepted
/// (`2^28` avoids any rounding at entry). After the last step the time samples are available
/// through [`q31_irfft_sample`] with value `sample·2^e` for the returned `Some(e)`; `1/N` is
/// included. Steps: `0` = normalize + tangling into bit-reversed order, `1..=log2(n/2)` =
/// inverse stages.
#[inline(always)]
pub(crate) fn q31_irfft_step<B: Words32 + ?Sized>(
    n: usize,
    step: usize,
    e_x: i32,
    buf: &mut B,
    st: &mut Q31FftState,
) -> Option<i32> {
    debug_assert!(q31_fft_supports(n) && step < q31_fft_steps(n));
    let m = n / 2;
    let last = q31_fft_steps(n) - 1;
    let zr0 = n + 2;
    let zi0 = zr0 + m;

    if step == 0 {
        let im0 = m + 1;
        let mut or_abs = 0u32;
        for k in 0..=m {
            or_abs |= buf.ld(k).unsigned_abs() | buf.ld(im0 + k).unsigned_abs();
        }
        if or_abs == 0 {
            st.zero = true;
            st.exp = ZERO_SPECTRUM_EXP;
        } else {
            st.zero = false;
            let sh = PAIR_LIMIT_BITS as i32 - bitlen(or_abs) as i32;
            st.or_abs = tangle(buf, m, sh);
            st.exp = e_x - sh;
        }
    } else if !st.zero {
        let s = headroom_shift(st.or_abs, STAGE_LIMIT_BITS);
        st.exp += s as i32;
        st.or_abs = cfft_stage(buf, zr0, zi0, m, 1 << (step - 1), true, s);
    }

    if step == last {
        if st.zero {
            for i in 0..n {
                buf.st(zr0 + i, 0);
            }
            Some(ZERO_SPECTRUM_EXP)
        } else {
            Some(st.exp - n.trailing_zeros() as i32)
        }
    } else {
        None
    }
}

/// Inverse of [`untangle`] (`Z(k) = 2E(k) + j·2O(k)`, so `x = IDFT_M(Z)/N`), written in
/// bit-reversed order to the packed-time area. Returns the OR of the written magnitudes.
#[inline(never)]
fn tangle<B: Words32 + ?Sized>(buf: &mut B, m: usize, sh: i32) -> u32 {
    let im0 = m + 1;
    let n = 2 * m;
    let zr0 = n + 2;
    let zi0 = zr0 + m;
    let lm = m.trailing_zeros();
    let mut or_abs = 0u32;

    let x0 = scale_to_fit(buf.ld(0), sh);
    let xm = scale_to_fit(buf.ld(m), sh);
    let (r, i) = (x0 + xm, x0 - xm);
    buf.st(zr0, r);
    buf.st(zi0, i);
    or_abs |= r.unsigned_abs() | i.unsigned_abs();

    let h = m / 2;
    let br_h = bit_reverse(h, lm);
    let (r, i) = (
        2 * scale_to_fit(buf.ld(h), sh),
        -2 * scale_to_fit(buf.ld(im0 + h), sh),
    );
    buf.st(zr0 + br_h, r);
    buf.st(zi0 + br_h, i);
    or_abs |= r.unsigned_abs() | i.unsigned_abs();

    let stride = MAX_Q31_FFT_SIZE / n;
    for k in 1..h {
        let mk = m - k;
        let xr1 = scale_to_fit(buf.ld(k), sh);
        let xi1 = scale_to_fit(buf.ld(im0 + k), sh);
        let xr2 = scale_to_fit(buf.ld(mk), sh);
        let xi2 = scale_to_fit(buf.ld(im0 + mk), sh);
        let (er, ei) = (xr1 + xr2, xi1 - xi2);
        let (dr, di) = (xr1 - xr2, xi1 + xi2);
        let (wc, ws) = twiddle_i32(k * stride);
        // conj(W_N^k) · D with conj(W_N^k) = wc + j·ws
        let or = cmul_part(wc, dr, ws, di);
        let oi = cmul_part(wc, di, -ws, dr);
        let (ar, ai, cr, ci) = (er - oi, ei + or, er + oi, -ei + or);
        let (bk, bmk) = (bit_reverse(k, lm), bit_reverse(mk, lm));
        buf.st(zr0 + bk, ar);
        buf.st(zi0 + bk, ai);
        buf.st(zr0 + bmk, cr);
        buf.st(zi0 + bmk, ci);
        or_abs |= ar.unsigned_abs() | ai.unsigned_abs() | cr.unsigned_abs() | ci.unsigned_abs();
    }
    count(h as u64, h.saturating_sub(1) as u64);
    or_abs
}

/// Time sample `i` (`0..n`) of a finished inverse transform (`x(2m) = Re z(m)`,
/// `x(2m+1) = Im z(m)`).
#[inline(always)]
pub(crate) fn q31_irfft_sample<B: Words32 + ?Sized>(buf: &B, n: usize, i: usize) -> i32 {
    let zr0 = n + 2;
    if i & 1 == 0 {
        buf.ld(zr0 + i / 2)
    } else {
        buf.ld(zr0 + n / 2 + i / 2)
    }
}

/// Quantizes a finished forward spectrum (`X(k) = mant·2^e`) to Q1.15 block-floating-point
/// storage with one round-to-nearest-even shift and saturation at `±32767`. Returns the stored
/// exponent `s_exp` with `X(k) ≈ out·2^(s_exp − 15)` (the convention of the Q1.15 engines), or
/// `ZERO_SPECTRUM_EXP` for an all-zero spectrum.
#[inline(never)]
pub(crate) fn q31_spectrum_to_q15<B: Words32 + ?Sized>(
    buf: &B,
    n: usize,
    e: i32,
    out_re: &mut [i16],
    out_im: &mut [i16],
) -> i32 {
    let m = n / 2;
    let im0 = m + 1;
    let mut or_abs = 0u32;
    for k in 0..=m {
        or_abs |= buf.ld(k).unsigned_abs() | buf.ld(im0 + k).unsigned_abs();
    }
    if or_abs == 0 {
        out_re[..=m].fill(0);
        out_im[..=m].fill(0);
        return ZERO_SPECTRUM_EXP;
    }
    let sh = 15 - bitlen(or_abs) as i32;
    let q = |v: i32| -> i16 {
        let r = if sh >= 0 {
            (v as i64) << sh
        } else {
            round_shift_even_i64(v as i64, sh.unsigned_abs())
        };
        r.clamp(-(i16::MAX as i64), i16::MAX as i64) as i16
    };
    for k in 0..=m {
        out_re[k] = q(buf.ld(k));
        out_im[k] = q(buf.ld(im0 + k));
    }
    e - sh + 15
}

/// Converts the per-bin `i64` accumulators of a region (`acc_re` in `i64` slots `0..bins`,
/// `acc_im` in slots `bins..2·bins`) **in place** into the spectrum layout of
/// [`q31_irfft_step`] (`i32` word `k` / `bins + k`), normalized to `|mant| <= 2^28` with one
/// round-to-nearest-even shift. Every `i32` word written lies at or before the `i64` slot it
/// comes from, so no unread accumulator is overwritten. `acc` values are `acc·2^acc_exp`;
/// returns the spectrum exponent, or `None` if all accumulators are zero (the region then reads
/// as a zero spectrum).
pub(crate) fn acc_i64_to_q31_spectrum_in_place<B: Words64 + ?Sized>(
    ws: &mut B,
    bins: usize,
    acc_exp: i32,
) -> Option<i32> {
    let mut or_abs = 0u64;
    for k in 0..2 * bins {
        or_abs |= ws.ld64(k).unsigned_abs();
    }
    if or_abs == 0 {
        return None;
    }
    let sh = PAIR_LIMIT_BITS as i32 - bitlen64(or_abs) as i32;
    let conv = |v: i64| -> i32 {
        if sh >= 0 {
            (v << sh) as i32
        } else {
            round_shift_even_i64(v, sh.unsigned_abs()) as i32
        }
    };
    for k in 0..2 * bins {
        let v = conv(ws.ld64(k));
        ws.st(k, v);
    }
    Some(acc_exp - sh)
}

/// Quantizes a finished forward spectrum (`X(k) = mant·2^e`) to `i32` block-floating-point
/// storage with `|out| <= 2^bits` (one round-to-nearest-even shift, `bits <= 30`). Returns `s`
/// with `X(k) ≈ out·2^s`, or `ZERO_SPECTRUM_EXP` for an all-zero spectrum.
#[cfg(feature = "fixed-q31")]
#[inline(never)]
pub(crate) fn q31_spectrum_to_i32<B: Words32 + ?Sized>(
    buf: &B,
    n: usize,
    e: i32,
    bits: u32,
    out_re: &mut [i32],
    out_im: &mut [i32],
) -> i32 {
    let m = n / 2;
    let im0 = m + 1;
    let mut or_abs = 0u32;
    for k in 0..=m {
        or_abs |= buf.ld(k).unsigned_abs() | buf.ld(im0 + k).unsigned_abs();
    }
    if or_abs == 0 {
        out_re[..=m].fill(0);
        out_im[..=m].fill(0);
        return ZERO_SPECTRUM_EXP;
    }
    let sh = bits as i32 - bitlen(or_abs) as i32;
    for k in 0..=m {
        out_re[k] = scale_to_fit(buf.ld(k), sh);
        out_im[k] = scale_to_fit(buf.ld(im0 + k), sh);
    }
    e - sh
}

/// `RNE(v·2^sh)` saturated to `[min, max]` (`sh` of either sign), reporting saturation.
#[inline(always)]
pub(crate) fn scale_rne_sat(v: i32, sh: i32, min: i64, max: i64) -> (i64, bool) {
    let r = if sh >= 0 {
        if v == 0 {
            0
        } else if sh >= 32 {
            if v > 0 { i64::MAX } else { i64::MIN }
        } else {
            (v as i64) << sh
        }
    } else if sh <= -40 {
        0
    } else {
        round_shift_even_i64(v as i64, sh.unsigned_abs())
    };
    if r > max {
        (max, true)
    } else if r < min {
        (min, true)
    } else {
        (r, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn dft(x: &[f64]) -> Vec<(f64, f64)> {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                let (mut r, mut i) = (0.0, 0.0);
                for (t, &v) in x.iter().enumerate() {
                    let a = -2.0 * core::f64::consts::PI * ((k * t) % n) as f64 / n as f64;
                    r += v * a.cos();
                    i += v * a.sin();
                }
                (r, i)
            })
            .collect()
    }

    fn snr_db(sig: f64, err: f64) -> f64 {
        10.0 * (sig / err.max(1e-300)).log10()
    }

    fn rng(seed: &mut u64) -> f64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn forward<B: Words32 + ?Sized>(x: &[i32], e_in: i32, buf: &mut B) -> i32 {
        let n = x.len();
        let mut st = Q31FftState::new();
        let mut out = None;
        for step in 0..q31_fft_steps(n) {
            out = q31_rfft_step(n, step, |i| x[i], e_in, buf, &mut st);
        }
        out.unwrap()
    }

    fn inverse<B: Words32 + ?Sized>(n: usize, e: i32, buf: &mut B) -> i32 {
        let mut st = Q31FftState::new();
        let mut out = None;
        for step in 0..q31_fft_steps(n) {
            out = q31_irfft_step(n, step, e, buf, &mut st);
        }
        out.unwrap()
    }

    /// f64 DFT oracle: forward SNR and forward+inverse roundtrip SNR >= 120 dB for every
    /// supported size, at full scale (Q1.31 input), at -60 dBFS and for a Q1.15 input, with both
    /// buffer representations.
    #[test]
    fn oracle_forward_and_roundtrip_snr_at_least_120_db() {
        let mut seed = 0x5EED_1234_u64;
        let mut n = Q31_FFT_MIN;
        while n <= Q31_FFT_MAX {
            for (amp, q15) in [(1.0f64, false), (1e-3, false), (0.5, true)] {
                let x: Vec<i32> = (0..n)
                    .map(|_| {
                        let v = rng(&mut seed) * amp;
                        if q15 {
                            ((v * 32768.0).round() as i32).clamp(-32768, 32767)
                        } else {
                            (v * 2_147_483_647.0).round() as i32
                        }
                    })
                    .collect();
                let e_in = if q15 { -15 } else { -31 };
                let xf: Vec<f64> = x.iter().map(|&v| v as f64 * (2.0f64).powi(e_in)).collect();
                let reference = dft(&xf);

                let mut buf32 = vec![0i32; q31_irfft_words(n)];
                let mut buf16 = vec![0i16; 2 * q31_irfft_words(n)];
                let e32 = forward(&x, e_in, &mut buf32[..]);
                let e16 = forward(&x, e_in, &mut buf16[..]);
                assert_eq!(e32, e16);
                let (mut s, mut err) = (0.0, 0.0);
                for (k, &(r, i)) in reference.iter().enumerate() {
                    let scale = (2.0f64).powi(e32);
                    let gr = buf32[k] as f64 * scale;
                    let gi = buf32[n / 2 + 1 + k] as f64 * scale;
                    assert_eq!(buf32[k], buf16[..].ld(k));
                    assert_eq!(buf32[n / 2 + 1 + k], buf16[..].ld(n / 2 + 1 + k));
                    s += r * r + i * i;
                    err += (r - gr).powi(2) + (i - gi).powi(2);
                }
                let fwd = snr_db(s, err);
                let et = inverse(n, e32, &mut buf32[..]);
                let et16 = inverse(n, e16, &mut buf16[..]);
                assert_eq!(et, et16);
                let (mut s, mut err) = (0.0, 0.0);
                for (t, &v) in xf.iter().enumerate() {
                    let got = q31_irfft_sample(&buf32[..], n, t) as f64 * (2.0f64).powi(et);
                    assert_eq!(
                        q31_irfft_sample(&buf32[..], n, t),
                        q31_irfft_sample(&buf16[..], n, t)
                    );
                    s += v * v;
                    err += (v - got).powi(2);
                }
                let rt = snr_db(s, err);
                std::println!(
                    "fft_q31_bfp N={n:5} amp={amp:7.0e} q15_in={q15}: forward SNR {fwd:7.2} dB, roundtrip SNR {rt:7.2} dB"
                );
                assert!(fwd >= 120.0, "N={n} forward SNR {fwd:.2}");
                assert!(rt >= 120.0, "N={n} roundtrip SNR {rt:.2}");
            }
            n *= 2;
        }
    }

    #[test]
    fn extremes_do_not_overflow_and_zero_is_exact() {
        for n in [64usize, 256, 2048] {
            // Full-scale DC, Nyquist alternation and i32::MIN impulses.
            for pat in 0..3 {
                let x: Vec<i32> = (0..n)
                    .map(|t| match pat {
                        0 => i32::MIN,
                        1 => {
                            if t % 2 == 0 {
                                i32::MAX
                            } else {
                                i32::MIN
                            }
                        }
                        _ => {
                            if t == 0 {
                                i32::MIN
                            } else {
                                0
                            }
                        }
                    })
                    .collect();
                let xf: Vec<f64> = x.iter().map(|&v| v as f64 / 2_147_483_648.0).collect();
                let mut buf = vec![0i32; q31_irfft_words(n)];
                let e = forward(&x, -31, &mut buf[..]);
                let reference = dft(&xf);
                let scale = (2.0f64).powi(e);
                for (k, &(r, i)) in reference.iter().enumerate() {
                    assert!(
                        (buf[k] as f64 * scale - r).abs() <= 1e-6 * n as f64,
                        "{pat} {k}"
                    );
                    assert!((buf[n / 2 + 1 + k] as f64 * scale - i).abs() <= 1e-6 * n as f64);
                }
                let et = inverse(n, e, &mut buf[..]);
                for (t, &v) in xf.iter().enumerate() {
                    let got = q31_irfft_sample(&buf[..], n, t) as f64 * (2.0f64).powi(et);
                    assert!((got - v).abs() < 1e-8, "pat={pat} n={n} t={t}");
                }
            }
            let mut buf = vec![7i32; q31_irfft_words(n)];
            let zero = vec![0i32; n];
            assert_eq!(forward(&zero, -15, &mut buf[..]), ZERO_SPECTRUM_EXP);
            assert!(buf[..q31_rfft_words(n)].iter().all(|&v| v == 0));
            assert_eq!(inverse(n, 0, &mut buf[..]), ZERO_SPECTRUM_EXP);
            assert!((0..n).all(|t| q31_irfft_sample(&buf[..], n, t) == 0));
        }
    }

    /// Deterministic operation count: the packed `N/2` transform plus untangling performs half
    /// the butterfly-equivalent operations of the former `N`-point complex transform of the
    /// same real data (and slightly fewer than half the complex multiplications).
    #[test]
    fn real_packing_halves_the_operation_count() {
        for lg in 6..=11u32 {
            let n = 1usize << lg;
            let x: Vec<i32> = (0..n as i32)
                .map(|t| (t * 7919) % 30_011 - 15_000)
                .collect();
            let mut buf = vec![0i32; q31_irfft_words(n)];
            let _ = opcount::take();
            let e = forward(&x, -15, &mut buf[..]);
            let (rf_pairs, rf_cmul) = opcount::take();
            let _ = inverse(n, e, &mut buf[..]);
            let (ri_pairs, ri_cmul) = opcount::take();

            // Baseline: N-point complex radix-2 FFT of x + j0 with the same stage kernel.
            let mut cre = vec![0i32; 2 * n];
            for (k, v) in cre[..n].iter_mut().enumerate() {
                *v = x[bit_reverse(k, lg)] << 14;
            }
            let mut half = 1;
            let mut or = 0;
            while half < n {
                let s = headroom_shift(or, STAGE_LIMIT_BITS);
                or = cfft_stage(&mut cre[..], 0, n, n, half, false, s);
                half <<= 1;
            }
            let (c_pairs, c_cmul) = opcount::take();
            std::println!(
                "N={n:5}: complex-N baseline {c_pairs} butterflies / {c_cmul} cmul; \
                 packed RFFT {rf_pairs} / {rf_cmul}; packed IRFFT {ri_pairs} / {ri_cmul}; \
                 ratio {:.3}",
                c_pairs as f64 / rf_pairs as f64
            );
            assert_eq!(c_pairs, (n / 2 * lg as usize) as u64);
            assert_eq!(rf_pairs * 2, c_pairs);
            assert_eq!(ri_pairs * 2, c_pairs);
            assert!(rf_cmul * 2 <= c_cmul && ri_cmul * 2 <= c_cmul);
        }
    }

    #[test]
    fn q15_storage_and_accumulator_helpers() {
        let n = 128;
        let x: Vec<i32> = (0..n as i32).map(|t| ((t * 37) % 200) - 100).collect();
        let mut buf = vec![0i16; 2 * q31_irfft_words(n)];
        let e = forward(&x, -15, &mut buf[..]);
        let (mut re, mut im) = (vec![0i16; n / 2 + 1], vec![0i16; n / 2 + 1]);
        let se = q31_spectrum_to_q15(&buf[..], n, e, &mut re, &mut im);
        let peak = re
            .iter()
            .chain(&im)
            .map(|v| v.unsigned_abs())
            .max()
            .unwrap();
        assert!((16_384..=32_767).contains(&peak), "peak {peak}");
        for (k, &r) in re.iter().enumerate() {
            let full = buf[..].ld(k) as f64 * (2.0f64).powi(e);
            let stored = r as f64 * (2.0f64).powi(se - 15);
            assert!((full - stored).abs() <= (2.0f64).powi(se - 15) * 0.5 + 1e-12);
        }
        // Accumulator layout round trip: i64 acc -> in-place i32 spectrum.
        let bins = n / 2 + 1;
        let mut ws = vec![0i16; 8 * bins];
        for k in 0..bins {
            crate::fixed::block_float::store_i64_to_i16(&mut ws, 4 * k, (k as i64 - 20) << 33);
            crate::fixed::block_float::store_i64_to_i16(&mut ws, 4 * (bins + k), -(k as i64) << 30);
        }
        let ex = acc_i64_to_q31_spectrum_in_place(&mut ws[..], bins, -10).unwrap();
        for k in 0..bins {
            let r = ws[..].ld(k) as f64 * (2.0f64).powi(ex);
            let i = ws[..].ld(bins + k) as f64 * (2.0f64).powi(ex);
            assert_eq!(r, ((k as f64 - 20.0) * (2.0f64).powi(33 - 10)));
            assert_eq!(i, -(k as f64) * (2.0f64).powi(30 - 10));
        }
        assert_eq!(scale_rne_sat(3, -1, -10, 10), (2, false));
        assert_eq!(scale_rne_sat(5, -1, -10, 10), (2, false));
        assert_eq!(scale_rne_sat(-5, -1, -10, 10), (-2, false));
        assert_eq!(scale_rne_sat(6, 1, -10, 10), (10, true));
        assert_eq!(scale_rne_sat(-1, 40, -10, 10), (-10, true));
    }
}
