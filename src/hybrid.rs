//! Zero-latency Hybrid FIR Convolver (`HybridConvolver`).
//!
//! Combines:
//! 1. A time-domain [`DirectConvolver`] covering the IR head `h[0..head_len]`.
//! 2. A frequency-domain [`PartitionedConvolver`] covering the IR tail `h[head_len..]`
//!    with partition block size `tail_block_size == head_len`.
//!
//! ## Exact Alignment & Latency Analysis
//! For `head_len == B_tail`:
//! - The tail contribution at sample `n` is `y_tail[n] = sum_{j>=0} h[B_tail + j] * x[n - B_tail - j]`.
//! - Therefore, the entire tail output block for `n in [m*B_tail, (m+1)*B_tail)` depends
//!   strictly on input samples `x[0 .. m*B_tail)`, all of which have already arrived by
//!   the end of block `m - 1`.
//! - While streaming host blocks of any size (e.g., `B_host = 32` while `B_tail = 128`),
//!   `HybridConvolver`:
//!   1. Computes the direct head response immediately.
//!   2. Adds the pre-computed `tail_out_buf` slice for the current offset.
//!   3. Copies the incoming samples into `tail_in_buf`.
//!   4. Whenever `tail_in_buf` reaches `B_tail` samples, runs one `PartitionedConvolver`
//!      step to populate `tail_out_buf` for the next `B_tail` samples.
//! - Result: **0 samples of algorithmic latency**, exact sample-by-sample temporal
//!   alignment with the original FIR, and zero boundary discontinuities.
//! - Real-time caveat: Every `B_tail / B_host` host callbacks, the tail FFT triggers
//!   synchronously inside `process_block`. Worst-case single-callback time must
//!   therefore accommodate both the direct head and one tail FFT step.

use crate::direct::{DirectConfig, DirectConvolver};
use crate::error::ConvError;
use crate::fft::{FftBackend, FftSize, MicroFftBackend};
use crate::partitioned::{PartitionedConfig, PartitionedConvolver};
use crate::simd_kernels::flush_tiny;
use crate::{Capabilities, EngineKind, SimdBackend};

/// Configuration for [`HybridConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridConfig {
    /// Length of the direct FIR head in samples. Must equal `tail_block_size`
    /// (power of two in `32..=8192`) to guarantee zero-latency alignment.
    pub head_len: usize,
    /// Partition block size `B_tail` for the FFT tail (power of two in `32..=8192`).
    pub tail_block_size: usize,
    /// Maximum supported total impulse response length in samples.
    pub max_ir_len: usize,
}

/// Checked workspace requirements for [`HybridConvolver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridMemoryRequirements {
    /// `f32` elements for the direct head convolver.
    pub direct_f32: usize,
    /// `f32` elements for the partitioned tail convolver (`0` if `max_ir_len <= head_len`).
    pub partitioned_f32: usize,
    /// `f32` elements for tail input/output staging buffers (`2 * tail_block_size`).
    pub staging_f32: usize,
    /// Total `f32` elements required in caller workspace.
    pub total_f32: usize,
    /// Total workspace size in bytes.
    pub total_bytes: usize,
}

/// Zero-allocation, `no_std` hybrid direct + partitioned FFT convolver.
pub struct HybridConvolver<'a, F: FftBackend = MicroFftBackend> {
    direct: DirectConvolver<'a>,
    tail: Option<PartitionedConvolver<'a, F>>,
    tail_in_buf: &'a mut [f32],
    tail_out_buf: &'a mut [f32],
    tail_pos: usize,
    tail_block_size: usize,
    tail_spreading: bool,
    spread_p: usize,
    acc_primed: bool,
    max_ir_len: usize,
    active_ir_len: usize,
    fft_name: &'static str,
    fft_is_simd: bool,
    fft_size: usize,
}

impl<'a, F: FftBackend> HybridConvolver<'a, F> {
    /// Computes checked workspace requirements for `config` and `fft`.
    pub fn memory_requirements(
        config: HybridConfig,
        fft: &F,
    ) -> Result<HybridMemoryRequirements, ConvError> {
        if config.max_ir_len == 0 {
            return Err(ConvError::EmptyIr);
        }
        let fft_size = FftSize::from_block_size(config.tail_block_size)?;
        if !fft.supports_size(fft_size) {
            return Err(ConvError::UnsupportedFftSize {
                fft_size: fft_size.len(),
            });
        }
        if config.head_len != config.tail_block_size {
            return Err(ConvError::InvalidHybridConfig {
                head_len: config.head_len,
                tail_block_size: config.tail_block_size,
            });
        }

        let direct_cap = config.head_len.min(config.max_ir_len);
        let direct_req = DirectConvolver::memory_requirements(DirectConfig {
            max_ir_len: direct_cap,
        })?;

        let tail_max = config.max_ir_len.saturating_sub(config.head_len);
        let partitioned_f32 = if tail_max > 0 {
            PartitionedConvolver::memory_requirements(
                PartitionedConfig {
                    block_size: config.tail_block_size,
                    max_ir_len: tail_max,
                },
                fft,
            )?
            .total_f32
        } else {
            0
        };

        let staging_f32 = config
            .tail_block_size
            .checked_mul(2)
            .ok_or(ConvError::ArithmeticOverflow)?;

        let total_f32 = direct_req
            .total_f32
            .checked_add(partitioned_f32)
            .and_then(|v| v.checked_add(staging_f32))
            .ok_or(ConvError::ArithmeticOverflow)?;
        let total_bytes = total_f32
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(ConvError::ArithmeticOverflow)?;

        Ok(HybridMemoryRequirements {
            direct_f32: direct_req.total_f32,
            partitioned_f32,
            staging_f32,
            total_f32,
            total_bytes,
        })
    }

    /// Validates inputs and initializes `HybridConvolver` inside caller-owned `workspace`.
    pub fn prepare(
        ir: &[f32],
        config: HybridConfig,
        workspace: &'a mut [f32],
        backend: SimdBackend,
        fft: F,
    ) -> Result<Self, ConvError> {
        let req = Self::memory_requirements(config, &fft)?;
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

        let fft_name = fft.name();
        let fft_is_simd = fft.is_simd_accelerated();
        let fft_size = config.tail_block_size * 2;

        let (direct_ws, rest) = workspace.split_at_mut(req.direct_f32);
        let (part_ws, rest) = rest.split_at_mut(req.partitioned_f32);
        let (tail_in_buf, rest) = rest.split_at_mut(config.tail_block_size);
        let (tail_out_buf, _) = rest.split_at_mut(config.tail_block_size);

        tail_in_buf.fill(0.0);
        tail_out_buf.fill(0.0);

        let head_cap = config.head_len.min(config.max_ir_len);
        let head_slice_len = ir.len().min(config.head_len);
        let direct = DirectConvolver::prepare(
            &ir[..head_slice_len],
            DirectConfig {
                max_ir_len: head_cap,
            },
            direct_ws,
            backend,
        )?;

        let tail = if ir.len() > config.head_len {
            let tail_max = config.max_ir_len - config.head_len;
            Some(PartitionedConvolver::prepare(
                &ir[config.head_len..],
                PartitionedConfig {
                    block_size: config.tail_block_size,
                    max_ir_len: tail_max,
                },
                part_ws,
                backend,
                fft,
            )?)
        } else {
            None
        };

        Ok(Self {
            direct,
            tail,
            tail_in_buf,
            tail_out_buf,
            tail_pos: 0,
            tail_block_size: config.tail_block_size,
            tail_spreading: false,
            spread_p: 1,
            acc_primed: false,
            max_ir_len: config.max_ir_len,
            active_ir_len: ir.len(),
            fft_name,
            fft_is_simd,
            fft_size,
        })
    }

    /// Convenience constructor equivalent to [`Self::prepare`] followed by
    /// [`Self::set_tail_spreading`].
    pub fn prepare_with_tail_spreading(
        ir: &[f32],
        config: HybridConfig,
        workspace: &'a mut [f32],
        backend: SimdBackend,
        fft: F,
        tail_spreading: bool,
    ) -> Result<Self, ConvError> {
        let mut conv = Self::prepare(ir, config, workspace, backend, fft)?;
        conv.set_tail_spreading(tail_spreading)?;
        Ok(conv)
    }

    /// Enables or disables zero-latency time-distributed FDL tail spreading.
    ///
    /// To prevent lost or duplicated partition contributions, mode changes are
    /// only permitted when no partial tail cycle is in progress (`tail_pos == 0`
    /// and `!acc_primed`, i.e., before streaming or immediately after [`Self::reset`]).
    /// Returns [`ConvError::InvalidStateTransition`] if called mid-cycle.
    #[inline]
    pub fn set_tail_spreading(&mut self, enabled: bool) -> Result<(), ConvError> {
        if self.tail_spreading == enabled {
            return Ok(());
        }
        if self.tail_pos != 0 || self.acc_primed {
            return Err(ConvError::InvalidStateTransition);
        }
        self.tail_spreading = enabled;
        self.spread_p = 1;
        self.acc_primed = false;
        Ok(())
    }

    /// Returns whether time-distributed FDL tail spreading is enabled.
    #[inline]
    pub const fn tail_spreading(&self) -> bool {
        self.tail_spreading
    }

    /// Processes an arbitrary-length host block (`input.len() == output.len()`).
    ///
    /// Supports host block sizes smaller than, equal to, or larger than `tail_block_size`
    /// (e.g., 32-sample host blocks with 64- or 128-sample FFT tail partitions).
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), ConvError> {
        if input.len() != output.len() {
            return Err(ConvError::MismatchedBufferLength {
                expected: input.len(),
                actual: output.len(),
            });
        }

        // 1. Direct FIR head produces exact immediate response for h[0..head_len].
        self.direct.process_block(input, output)?;

        // 2. If there is an active FFT tail, add staged tail output and feed tail FDL.
        if let Some(ref mut tail_conv) = self.tail {
            let total = input.len();
            let b_tail = self.tail_block_size;
            let mut processed = 0;

            while processed < total {
                let space = b_tail - self.tail_pos;
                let chunk = (total - processed).min(space);

                let mut i = 0;
                while i < chunk {
                    output[processed + i] =
                        flush_tiny(output[processed + i] + self.tail_out_buf[self.tail_pos + i]);
                    self.tail_in_buf[self.tail_pos + i] = flush_tiny(input[processed + i]);
                    i += 1;
                }

                self.tail_pos += chunk;
                processed += chunk;

                if !self.tail_spreading {
                    if self.tail_pos == b_tail {
                        tail_conv.process_block(self.tail_in_buf, self.tail_out_buf)?;
                        self.tail_pos = 0;
                    }
                } else {
                    let p_total = tail_conv.num_partitions();
                    if !self.acc_primed {
                        tail_conv.reset_accumulator();
                        self.spread_p = 1;
                        self.acc_primed = true;
                    }

                    if self.tail_pos < b_tail {
                        // Distribute p = 1..P evenly across the pre-completion sub-blocks.
                        if p_total > 1 {
                            let past = p_total - 1;
                            let denom = b_tail.saturating_sub(chunk).max(1);
                            let target_p = (1 + (past * self.tail_pos) / denom).min(p_total);
                            if target_p > self.spread_p {
                                tail_conv.accumulate_past_partitions(self.spread_p, target_p);
                                self.spread_p = target_p;
                            }
                        }
                    } else {
                        // Complete any remaining past partitions, then RFFT + CMAC(p=0) + IFFT.
                        if self.spread_p < p_total {
                            tail_conv.accumulate_past_partitions(self.spread_p, p_total);
                        }
                        tail_conv.finish_distributed_block(self.tail_in_buf, self.tail_out_buf)?;
                        self.tail_pos = 0;
                        self.spread_p = 1;
                        self.acc_primed = false;
                    }
                }
            }
        }

        Ok(())
    }

    /// Resets both direct head and partitioned tail state without deallocation.
    #[inline]
    pub fn reset(&mut self) {
        self.direct.reset();
        if let Some(ref mut tail_conv) = self.tail {
            tail_conv.reset();
        }
        self.tail_in_buf.fill(0.0);
        self.tail_out_buf.fill(0.0);
        self.tail_pos = 0;
        self.spread_p = 1;
        self.acc_primed = false;
    }

    /// Algorithmic latency in samples (always `0` because `head_len == tail_block_size`).
    #[inline]
    pub const fn latency_samples(&self) -> usize {
        0
    }

    /// Engine capabilities and runtime metadata.
    #[inline]
    pub fn capabilities(&self) -> Capabilities {
        let direct_caps = self.direct.capabilities();
        Capabilities {
            engine: EngineKind::Hybrid,
            simd_backend: direct_caps.simd_backend,
            simd_active: direct_caps.simd_active,
            fft_backend: self.fft_name,
            fft_is_simd: self.fft_is_simd,
            block_size: Some(self.tail_block_size),
            fft_size: Some(self.fft_size),
            max_ir_len: self.max_ir_len,
            active_ir_len: self.active_ir_len,
            latency_samples: 0,
        }
    }
}
