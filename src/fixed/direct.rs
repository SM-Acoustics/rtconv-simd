//! Pure Rust `#![no_std]` Q1.15 direct time-domain FIR convolver (`FixedQ15DirectConvolver`).
//!
//! Designed for FPU-less microcontrollers (such as ESP32-C6 / `riscv32imac-unknown-none-elf`),
//! using caller-supplied `&mut [i16]` memory, `i64` accumulation, explicit round-half-up
//! scaling (`(acc + 16384) >> 15`), and symmetric saturation with saturation telemetry.

use crate::ConvError;
use crate::fixed::q15::round_sat_q15_from_i64;

/// Configuration for [`FixedQ15DirectConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ15DirectConfig {
    /// Maximum supported impulse response length in taps.
    pub max_ir_len: usize,
}

/// Caller-supplied memory requirements for [`FixedQ15DirectConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ15DirectMemoryRequirements {
    /// Elements reserved for FIR coefficients (`i16`).
    pub coeffs_i16: usize,
    /// Elements reserved for the circular input history (`i16`).
    pub history_i16: usize,
    /// Total required `&mut [i16]` workspace elements.
    pub total_i16: usize,
    /// Total required workspace size in bytes (`total_i16 * 2`).
    pub total_bytes: usize,
}

/// Runtime metadata for [`FixedQ15DirectConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedQ15DirectCapabilities {
    /// Maximum configured impulse response length.
    pub max_ir_len: usize,
    /// Active impulse response length.
    pub active_ir_len: usize,
    /// Algorithmic latency in samples (always `0`).
    pub latency_samples: usize,
    /// Accumulator bit width (`64`).
    pub accumulator_bits: u8,
    /// Explicit rounding and saturation policy description.
    pub rounding_policy: &'static str,
}

/// Zero-latency, FPU-free Q1.15 direct time-domain FIR convolver.
pub struct FixedQ15DirectConvolver<'a> {
    coeffs: &'a mut [i16],
    history: &'a mut [i16],
    max_ir_len: usize,
    active_ir_len: usize,
    head: usize,
    saturation_count: u64,
}

impl<'a> FixedQ15DirectConvolver<'a> {
    /// Computes the exact `&mut [i16]` workspace requirements for `config`.
    pub fn memory_requirements(
        config: FixedQ15DirectConfig,
    ) -> Result<FixedQ15DirectMemoryRequirements, ConvError> {
        if config.max_ir_len == 0 {
            return Err(ConvError::EmptyIr);
        }
        let total_i16 = config
            .max_ir_len
            .checked_mul(2)
            .ok_or(ConvError::ArithmeticOverflow)?;
        let total_bytes = total_i16
            .checked_mul(core::mem::size_of::<i16>())
            .ok_or(ConvError::ArithmeticOverflow)?;
        Ok(FixedQ15DirectMemoryRequirements {
            coeffs_i16: config.max_ir_len,
            history_i16: config.max_ir_len,
            total_i16,
            total_bytes,
        })
    }

    /// Prepares a [`FixedQ15DirectConvolver`] using caller-supplied `&'a mut [i16]` memory.
    pub fn prepare(
        ir: &[i16],
        config: FixedQ15DirectConfig,
        workspace: &'a mut [i16],
    ) -> Result<Self, ConvError> {
        if ir.is_empty() {
            return Err(ConvError::EmptyIr);
        }
        if ir.len() > config.max_ir_len {
            return Err(ConvError::IrTooLong {
                ir_len: ir.len(),
                max_ir_len: config.max_ir_len,
            });
        }
        let req = Self::memory_requirements(config)?;
        if workspace.len() < req.total_i16 {
            return Err(ConvError::WorkspaceTooSmall {
                required: req.total_i16,
                provided: workspace.len(),
            });
        }

        let (coeffs, rest) = workspace.split_at_mut(config.max_ir_len);
        let history = &mut rest[..config.max_ir_len];
        coeffs.fill(0);
        coeffs[..ir.len()].copy_from_slice(ir);
        history.fill(0);

        Ok(Self {
            coeffs,
            history,
            max_ir_len: config.max_ir_len,
            active_ir_len: ir.len(),
            head: 0,
            saturation_count: 0,
        })
    }

    /// Processes a single Q1.15 sample (`i16`) with `i64` accumulation, round-half-up,
    /// and saturation.
    #[inline]
    pub fn process_sample(&mut self, x: i16) -> i16 {
        let len = self.max_ir_len;
        let active = self.active_ir_len;
        let head = self.head;
        self.history[head] = x;

        let first_seg = (head + 1).min(active);
        let mut acc = 0i64;
        for k in 0..first_seg {
            acc += (self.coeffs[k] as i64) * (self.history[head - k] as i64);
        }
        let rem = active - first_seg;
        if rem > 0 {
            let mut h_idx = len - 1;
            for &c in &self.coeffs[first_seg..active] {
                acc += (c as i64) * (self.history[h_idx] as i64);
                h_idx = h_idx.saturating_sub(1);
            }
        }

        self.head = if head + 1 == len { 0 } else { head + 1 };

        let (y, sat) = round_sat_q15_from_i64(acc);
        if sat {
            self.saturation_count = self.saturation_count.saturating_add(1);
        }
        y
    }

    /// Processes an arbitrary-length block of Q1.15 samples (`input.len() == output.len()`).
    pub fn process_block(&mut self, input: &[i16], output: &mut [i16]) -> Result<(), ConvError> {
        if input.len() != output.len() {
            return Err(ConvError::MismatchedBufferLength {
                expected: input.len(),
                actual: output.len(),
            });
        }
        for (&x, y) in input.iter().zip(output.iter_mut()) {
            *y = self.process_sample(x);
        }
        Ok(())
    }

    /// Clears the circular input history and resets the saturation counter while preserving
    /// the loaded FIR coefficients.
    pub fn reset(&mut self) {
        self.history.fill(0);
        self.head = 0;
        self.saturation_count = 0;
    }

    /// Algorithmic latency in samples (always `0`).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        0
    }

    /// Number of output samples that clipped to `[i16::MIN, i16::MAX]` since `prepare()` or `reset()`.
    #[inline]
    pub const fn saturation_count(&self) -> u64 {
        self.saturation_count
    }

    /// Runtime capabilities and numeric metadata.
    #[inline]
    pub const fn capabilities(&self) -> FixedQ15DirectCapabilities {
        FixedQ15DirectCapabilities {
            max_ir_len: self.max_ir_len,
            active_ir_len: self.active_ir_len,
            latency_samples: 0,
            accumulator_bits: 64,
            rounding_policy: "round-half-up ((acc + 16384) >> 15) with i16 saturation",
        }
    }
}
