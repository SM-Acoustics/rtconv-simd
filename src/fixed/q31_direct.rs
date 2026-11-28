//! Q1.31 direct time-domain FIR convolver (`FixedQ31DirectConvolver`) for 24/32-bit audio on
//! FPU-less MCUs (ESP32-C6 `riscv32imac`, Cortex-M0).
//!
//! `#![no_std]`, zero heap, `#![forbid(unsafe_code)]`, integer arithmetic only.
//!
//! ## Arithmetic
//! - Samples and taps are Q1.31 (`i32`); every product `x·h` is exact in `i64` (Q2.62, one
//!   `mul` + `mulh` on RV32IM) and is accumulated **exactly** in `i64`.
//! - Overflow is excluded at `prepare` from the L1 norm of the taps: with
//!   `‖h‖₁·2^31 = Σ|h_k|` (integer) and `|x| <= 2^31`, `|Σ x·h| <= 2^31·Σ|h_k|`, so the sum fits
//!   when `Σ|h_k| < 2^32` (`‖h‖₁ < 2`, every audio filter up to +6 dB gain). For larger gains
//!   each product is pre-shifted by `g = bitlen(Σ|h_k|) − 32` bits (arithmetic shift; the
//!   error is below `2^(g−31)·len` output LSB) and reported in the capabilities.
//! - The output is rounded **once** to Q1.31 with round-to-nearest-even and saturated
//!   symmetrically to `±(2^31 − 1)` (an exact `−1.0` result is clipped by one LSB and counted).
//!
//! ## Type I linear-phase folding
//! When the IR has odd length `L` and even symmetry `h[k] = h[L−1−k]` (FIR Type I, e.g. a
//! linear-phase crossover), and `‖h‖₁ < 2`, the pairs `x[n−k] + x[n−L+1+k]` are added in
//! `i64` before the multiplication, so a 255-tap filter costs 128 MACs per sample instead of
//! 255. The integer result is identical to the unfolded sum (bit-exact, not just `<= 1 LSB`).
//!
//! ## Memory and inner loop
//! The history is a linear double buffer (`2·L` words: each sample is written at `pos` and
//! `pos + L`), so the `L` newest samples are always the contiguous window
//! `hist[pos..pos + L]`. The MAC loops zip two contiguous slices (forward and reversed), with
//! no modulo and no per-tap branch.

use crate::ConvError;

/// Configuration for [`FixedQ31DirectConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ31DirectConfig {
    /// Maximum supported impulse response length in taps.
    pub max_ir_len: usize,
}

/// Caller-supplied `&mut [i32]` memory requirements for [`FixedQ31DirectConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ31DirectMemoryRequirements {
    /// Words reserved for the Q1.31 coefficients.
    pub coeffs_i32: usize,
    /// Words reserved for the linear double-buffered history (`2 * max_ir_len`).
    pub history_i32: usize,
    /// Total required `&mut [i32]` words.
    pub total_i32: usize,
    /// Total workspace bytes (`total_i32 * 4`).
    pub total_bytes: usize,
}

/// Symmetry exploited by a prepared [`FixedQ31DirectConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedQ31Symmetry {
    /// Plain `L` MACs per sample.
    None,
    /// FIR Type I (odd length, `h[k] = h[L−1−k]`): `(L + 1) / 2` MACs per sample.
    TypeIEven,
}

/// Runtime metadata for [`FixedQ31DirectConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ31DirectCapabilities {
    /// Maximum configured impulse response length.
    pub max_ir_len: usize,
    /// Active impulse response length.
    pub active_ir_len: usize,
    /// Algorithmic latency in samples (always `0`).
    pub latency_samples: usize,
    /// Symmetry folding in use.
    pub symmetry: FixedQ31Symmetry,
    /// Multiply-accumulates per output sample.
    pub macs_per_sample: usize,
    /// Product pre-shift `g` (`0` = exact accumulation, i.e. `‖h‖₁ < 2`).
    pub product_shift: u32,
}

/// Zero-latency, FPU-free Q1.31 direct-form FIR convolver.
pub struct FixedQ31DirectConvolver<'a> {
    coeffs: &'a mut [i32],
    history: &'a mut [i32],
    max_ir_len: usize,
    len: usize,
    pos: usize,
    symmetry: FixedQ31Symmetry,
    shift: u32,
    saturation_count: u64,
}

/// `RNE(acc / 2^s)` saturated symmetrically to `±(2^31 − 1)`, `1 <= s <= 62`.
#[inline(always)]
pub(crate) fn round_sat_q31_sym(acc: i64, s: u32) -> (i32, bool) {
    let half = 1i64 << (s - 1);
    let q = acc >> s;
    let rem = acc & ((1i64 << s) - 1);
    let r = if rem > half || (rem == half && (q & 1) != 0) {
        q + 1
    } else {
        q
    };
    if r > i32::MAX as i64 {
        (i32::MAX, true)
    } else if r < -(i32::MAX as i64) {
        (-i32::MAX, true)
    } else {
        (r as i32, false)
    }
}

/// Product pre-shift so that `Σ|x·h| >> g` fits `i64` for any Q1.31 input (taps supplied by
/// index).
#[inline]
pub(crate) fn q31_product_shift_with<F: Fn(usize) -> i32>(len: usize, tap: F) -> u32 {
    let l1: u64 = (0..len).map(|i| (tap(i) as i64).unsigned_abs()).sum();
    (64 - l1.leading_zeros()).saturating_sub(32)
}

impl<'a> FixedQ31DirectConvolver<'a> {
    /// Exact `&mut [i32]` workspace for `config` (`3 * max_ir_len` words), usable in `const`
    /// contexts (e.g. to size a `static` array).
    pub const fn memory_requirements(
        config: FixedQ31DirectConfig,
    ) -> Result<FixedQ31DirectMemoryRequirements, ConvError> {
        if config.max_ir_len == 0 {
            return Err(ConvError::EmptyIr);
        }
        let Some(history_i32) = config.max_ir_len.checked_mul(2) else {
            return Err(ConvError::ArithmeticOverflow);
        };
        let Some(total_i32) = history_i32.checked_add(config.max_ir_len) else {
            return Err(ConvError::ArithmeticOverflow);
        };
        let Some(total_bytes) = total_i32.checked_mul(4) else {
            return Err(ConvError::ArithmeticOverflow);
        };
        Ok(FixedQ31DirectMemoryRequirements {
            coeffs_i32: config.max_ir_len,
            history_i32,
            total_i32,
            total_bytes,
        })
    }

    /// Prepares the convolver, detecting FIR Type I symmetry automatically (folded only when
    /// it is exact, i.e. `‖h‖₁ < 2`).
    pub fn prepare(
        ir: &[i32],
        config: FixedQ31DirectConfig,
        workspace: &'a mut [i32],
    ) -> Result<Self, ConvError> {
        Self::prepare_impl(ir.len(), |i| ir[i], config, workspace, true)
    }

    /// Like [`Self::prepare`] with `len` taps supplied by index (`tap(i)`), e.g. decoded on the
    /// fly from a FIR artifact without an intermediate `&[i32]`.
    pub fn prepare_with<F: Fn(usize) -> i32>(
        len: usize,
        tap: F,
        config: FixedQ31DirectConfig,
        workspace: &'a mut [i32],
    ) -> Result<Self, ConvError> {
        Self::prepare_impl(len, tap, config, workspace, true)
    }

    /// Prepares the convolver without symmetry folding (`L` MACs per sample), e.g. as the
    /// reference for a folded instance.
    pub fn prepare_unfolded(
        ir: &[i32],
        config: FixedQ31DirectConfig,
        workspace: &'a mut [i32],
    ) -> Result<Self, ConvError> {
        Self::prepare_impl(ir.len(), |i| ir[i], config, workspace, false)
    }

    fn prepare_impl<F: Fn(usize) -> i32>(
        len: usize,
        tap: F,
        config: FixedQ31DirectConfig,
        workspace: &'a mut [i32],
        allow_fold: bool,
    ) -> Result<Self, ConvError> {
        if len == 0 {
            return Err(ConvError::EmptyIr);
        }
        if len > config.max_ir_len {
            return Err(ConvError::IrTooLong {
                ir_len: len,
                max_ir_len: config.max_ir_len,
            });
        }
        let req = Self::memory_requirements(config)?;
        if workspace.len() < req.total_i32 {
            return Err(ConvError::WorkspaceTooSmall {
                required: req.total_i32,
                provided: workspace.len(),
            });
        }
        let shift = q31_product_shift_with(len, &tap);
        let symmetric = len % 2 == 1 && (0..len / 2).all(|k| tap(k) == tap(len - 1 - k));
        let symmetry = if allow_fold && symmetric && len >= 3 && shift == 0 {
            FixedQ31Symmetry::TypeIEven
        } else {
            FixedQ31Symmetry::None
        };
        let (coeffs, rest) = workspace[..req.total_i32].split_at_mut(config.max_ir_len);
        let history = &mut rest[..req.history_i32];
        coeffs.fill(0);
        history.fill(0);
        let stored = match symmetry {
            FixedQ31Symmetry::TypeIEven => len / 2 + 1,
            FixedQ31Symmetry::None => len,
        };
        for (i, c) in coeffs[..stored].iter_mut().enumerate() {
            *c = tap(i);
        }
        Ok(Self {
            coeffs,
            history,
            max_ir_len: config.max_ir_len,
            len,
            pos: 0,
            symmetry,
            shift,
            saturation_count: 0,
        })
    }

    /// Exact `i64` dot product of the coefficients with the current window
    /// (`window[k] = x[n − k]`), scaled by `2^-shift`.
    #[inline(always)]
    fn dot(&self) -> i64 {
        let l = self.len;
        let w = &self.history[self.pos..self.pos + l];
        match self.symmetry {
            FixedQ31Symmetry::TypeIEven => {
                let m = l / 2;
                let h = &self.coeffs[..m];
                let mut acc = self.coeffs[m] as i64 * w[m] as i64;
                // Pairs x[n−k] + x[n−L+1+k]: forward head and reversed tail, both contiguous.
                for ((&c, &a), &b) in h.iter().zip(&w[..m]).zip(w[m + 1..].iter().rev()) {
                    acc += c as i64 * (a as i64 + b as i64);
                }
                acc
            }
            FixedQ31Symmetry::None => {
                let h = &self.coeffs[..l];
                let mut acc = 0i64;
                if self.shift == 0 {
                    let (hc, ht) = h.as_chunks::<4>();
                    let (wc, wt) = w.as_chunks::<4>();
                    for (c, x) in hc.iter().zip(wc) {
                        acc += c[0] as i64 * x[0] as i64
                            + c[1] as i64 * x[1] as i64
                            + c[2] as i64 * x[2] as i64
                            + c[3] as i64 * x[3] as i64;
                    }
                    for (&c, &x) in ht.iter().zip(wt) {
                        acc += c as i64 * x as i64;
                    }
                } else {
                    let g = self.shift;
                    for (&c, &x) in h.iter().zip(w) {
                        acc += (c as i64 * x as i64) >> g;
                    }
                }
                acc
            }
        }
    }

    /// Processes one Q1.31 sample.
    #[inline]
    pub fn process_sample(&mut self, x: i32) -> i32 {
        let l = self.len;
        self.pos = if self.pos == 0 { l - 1 } else { self.pos - 1 };
        self.history[self.pos] = x;
        self.history[self.pos + l] = x;
        let (y, sat) = round_sat_q31_sym(self.dot(), 31 - self.shift);
        self.saturation_count += sat as u64;
        y
    }

    /// Processes `input` into `output` (equal lengths, any block size).
    pub fn process_block(&mut self, input: &[i32], output: &mut [i32]) -> Result<(), ConvError> {
        if input.len() != output.len() {
            return Err(ConvError::MismatchedBufferLength {
                expected: input.len(),
                actual: output.len(),
            });
        }
        for (y, &x) in output.iter_mut().zip(input) {
            *y = self.process_sample(x);
        }
        Ok(())
    }

    /// Clears the history and the saturation counter, keeping the filter.
    pub fn reset(&mut self) {
        self.history.fill(0);
        self.pos = 0;
        self.saturation_count = 0;
    }

    /// Algorithmic latency in samples (always `0`).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        0
    }

    /// Output samples clipped by saturation since `prepare` or [`Self::reset`].
    #[inline]
    pub const fn saturation_count(&self) -> u64 {
        self.saturation_count
    }

    /// Runtime metadata.
    pub const fn capabilities(&self) -> FixedQ31DirectCapabilities {
        FixedQ31DirectCapabilities {
            max_ir_len: self.max_ir_len,
            active_ir_len: self.len,
            latency_samples: 0,
            symmetry: self.symmetry,
            macs_per_sample: match self.symmetry {
                FixedQ31Symmetry::TypeIEven => self.len / 2 + 1,
                FixedQ31Symmetry::None => self.len,
            },
            product_shift: self.shift,
        }
    }
}
