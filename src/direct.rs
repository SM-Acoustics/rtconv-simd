//! Time-domain direct FIR convolution engine (`DirectConvolver`).
//!
//! Computes `y[n] = sum_{k=0}^{L-1} h[k] * x[n - k]` using a pre-allocated
//! reverse-indexed circular buffer so both the impulse response and history
//! segments are traversed in forward unit-stride order for SIMD dot products.

use crate::error::ConvError;
use crate::simd_kernels::{
    copy_flush_tiny, dot_product_scalar, dot_two_segments_simd, fir_block_scalar, fir_block_simd,
    flush_tiny,
};
use crate::{Capabilities, EngineKind, SimdBackend};

/// Configuration for [`DirectConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectConfig {
    /// Maximum supported FIR length in taps.
    pub max_ir_len: usize,
}

/// Checked memory requirements for [`DirectConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectMemoryRequirements {
    /// Elements reserved for impulse response coefficients.
    pub ir_len: usize,
    /// Elements reserved for the circular sample history buffer.
    pub history_len: usize,
    /// Total `f32` workspace elements required.
    pub total_f32: usize,
    /// Total workspace size in bytes.
    pub total_bytes: usize,
}

/// Zero-allocation, `no_std` time-domain FIR convolver.
pub struct DirectConvolver<'a> {
    ir: &'a mut [f32],
    history: &'a mut [f32],
    ir_len: usize,
    write_pos: usize,
    backend: SimdBackend,
}

impl<'a> DirectConvolver<'a> {
    /// Computes workspace requirements using checked arithmetic.
    pub fn memory_requirements(
        config: DirectConfig,
    ) -> Result<DirectMemoryRequirements, ConvError> {
        if config.max_ir_len == 0 {
            return Err(ConvError::EmptyIr);
        }
        let ir_len = config.max_ir_len;
        let history_len = config.max_ir_len;
        let total_f32 = ir_len
            .checked_add(history_len)
            .ok_or(ConvError::ArithmeticOverflow)?;
        let total_bytes = total_f32
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(ConvError::ArithmeticOverflow)?;
        Ok(DirectMemoryRequirements {
            ir_len,
            history_len,
            total_f32,
            total_bytes,
        })
    }

    /// Validates inputs and initializes a `DirectConvolver` inside caller-owned `workspace`.
    ///
    /// Preserves the exact coefficients, order, amplitude, phase, and delay of `ir`, except that
    /// magnitudes below [`crate::simd_kernels::DENORMAL_GUARD`] (`2^-63`) are flushed to zero.
    pub fn prepare(
        ir: &[f32],
        config: DirectConfig,
        workspace: &'a mut [f32],
        backend: SimdBackend,
    ) -> Result<Self, ConvError> {
        let req = Self::memory_requirements(config)?;
        if ir.is_empty() {
            return Err(ConvError::EmptyIr);
        }
        if ir.len() > config.max_ir_len {
            return Err(ConvError::IrTooLong {
                ir_len: ir.len(),
                max_ir_len: config.max_ir_len,
            });
        }
        for (idx, &coeff) in ir.iter().enumerate() {
            if !coeff.is_finite() {
                return Err(ConvError::NonFiniteCoefficient { index: idx });
            }
        }
        if workspace.len() < req.total_f32 {
            return Err(ConvError::WorkspaceTooSmall {
                required: req.total_f32,
                provided: workspace.len(),
            });
        }

        let (ir_buf, rest) = workspace.split_at_mut(req.ir_len);
        let (history_buf, _) = rest.split_at_mut(req.history_len);

        let ir_len = ir.len();
        // Coefficients below `DENORMAL_GUARD` are flushed (see `simd_kernels::DENORMAL_GUARD`).
        copy_flush_tiny(&mut ir_buf[..ir_len], ir);
        ir_buf[ir_len..].fill(0.0);
        history_buf.fill(0.0);

        Ok(Self {
            ir: ir_buf,
            history: history_buf,
            ir_len,
            write_pos: 0,
            backend,
        })
    }

    /// Processes a single audio sample without allocation or panics.
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let ir_len = self.ir_len;
        let pos = if self.write_pos == 0 {
            ir_len - 1
        } else {
            self.write_pos - 1
        };
        self.write_pos = pos;
        self.history[pos] = flush_tiny(input);

        let first_len = ir_len - pos;
        let ir_active = &self.ir[..ir_len];
        let hist_active = &self.history[..ir_len];

        let y = if self.backend.is_scalar() {
            dot_product_scalar(&ir_active[..first_len], &hist_active[pos..ir_len])
                + dot_product_scalar(&ir_active[first_len..ir_len], &hist_active[..pos])
        } else {
            dot_two_segments_simd(
                self.backend.level(),
                &ir_active[..first_len],
                &hist_active[pos..ir_len],
                &ir_active[first_len..ir_len],
                &hist_active[..pos],
            )
        };
        flush_tiny(y)
    }

    /// Processes a block of samples using a single SIMD dispatch at block entry.
    #[inline]
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), ConvError> {
        if input.len() != output.len() {
            return Err(ConvError::MismatchedBufferLength {
                expected: input.len(),
                actual: output.len(),
            });
        }
        let ir_len = self.ir_len;
        let ir_active = &self.ir[..ir_len];
        let hist_active = &mut self.history[..ir_len];

        if self.backend.is_scalar() {
            fir_block_scalar(ir_active, hist_active, &mut self.write_pos, input, output);
        } else {
            fir_block_simd(
                self.backend.level(),
                ir_active,
                hist_active,
                &mut self.write_pos,
                input,
                output,
            );
        }
        Ok(())
    }

    /// Clears the circular delay line without deallocating memory.
    #[inline]
    pub fn reset(&mut self) {
        self.history[..self.ir_len].fill(0.0);
        self.write_pos = 0;
    }

    /// Algorithmic latency in samples (always `0` for direct FIR convolution).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        0
    }

    /// Active impulse response length in taps.
    #[inline]
    pub const fn ir_len(&self) -> usize {
        self.ir_len
    }

    /// Engine capabilities and runtime metadata.
    #[inline]
    pub fn capabilities(&self) -> Capabilities {
        Capabilities {
            engine: EngineKind::Direct,
            simd_backend: self.backend.name(),
            simd_active: self.backend.is_simd_active(),
            fft_backend: "none",
            fft_is_simd: false,
            block_size: None,
            fft_size: None,
            max_ir_len: self.ir.len(),
            active_ir_len: self.ir_len,
            latency_samples: 0,
        }
    }
}
