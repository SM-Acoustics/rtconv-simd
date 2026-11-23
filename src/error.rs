//! Typed errors for configuration, preparation, and boundary validation.

use core::fmt;

/// Error returned during convolver configuration, preparation, or contract-checked calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvError {
    /// Impulse response slice is empty.
    EmptyIr,
    /// Impulse response length exceeds configured `max_ir_len`.
    IrTooLong {
        /// Actual impulse response length.
        ir_len: usize,
        /// Configured maximum capacity.
        max_ir_len: usize,
    },
    /// Impulse response contains `NaN` or infinite values.
    NonFiniteCoefficient {
        /// Index of the first non-finite sample.
        index: usize,
    },
    /// Block size is invalid or unsupported for the selected engine.
    InvalidBlockSize {
        /// Requested block size.
        block_size: usize,
    },
    /// FFT size is unsupported by the selected [`crate::FftBackend`].
    UnsupportedFftSize {
        /// Requested FFT size in samples.
        fft_size: usize,
    },
    /// Caller-provided workspace slice is smaller than required.
    WorkspaceTooSmall {
        /// Minimum required `f32` elements.
        required: usize,
        /// Provided `f32` elements.
        provided: usize,
    },
    /// Input and output buffer lengths do not match the engine contract.
    MismatchedBufferLength {
        /// Expected length (or input length).
        expected: usize,
        /// Provided length (or output length).
        actual: usize,
    },
    /// Checked arithmetic overflow while calculating memory requirements.
    ArithmeticOverflow,
    /// Hybrid configuration split (`head_len`, `tail_block_size`) is invalid.
    InvalidHybridConfig {
        /// Requested head FIR length.
        head_len: usize,
        /// Requested tail partition block size.
        tail_block_size: usize,
    },
    /// Requested SIMD backend is not available on this target or feature set.
    BackendUnavailable,
    /// Attempted to change runtime processing mode (e.g. hybrid tail spreading)
    /// while a partial cycle is in progress; call before streaming or after `reset()`.
    InvalidStateTransition,
    /// Non-uniform partitioned configuration violates causality, contiguity, stage count,
    /// or per-callback deadline/budget constraints.
    InvalidNonUniformConfig,
}

impl fmt::Display for ConvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIr => write!(f, "impulse response must not be empty"),
            Self::IrTooLong { ir_len, max_ir_len } => {
                write!(
                    f,
                    "impulse response length ({ir_len}) exceeds configured max_ir_len ({max_ir_len})"
                )
            }
            Self::NonFiniteCoefficient { index } => {
                write!(
                    f,
                    "non-finite impulse response coefficient at index {index}"
                )
            }
            Self::InvalidBlockSize { block_size } => {
                write!(f, "invalid or unsupported block size: {block_size}")
            }
            Self::UnsupportedFftSize { fft_size } => {
                write!(f, "FFT size {fft_size} is not supported by the backend")
            }
            Self::WorkspaceTooSmall { required, provided } => {
                write!(
                    f,
                    "workspace too small: required {required} f32 elements, provided {provided}"
                )
            }
            Self::MismatchedBufferLength { expected, actual } => {
                write!(
                    f,
                    "mismatched buffer length: expected {expected}, got {actual}"
                )
            }
            Self::ArithmeticOverflow => write!(f, "arithmetic overflow in memory calculation"),
            Self::InvalidHybridConfig {
                head_len,
                tail_block_size,
            } => {
                write!(
                    f,
                    "invalid hybrid config: head_len ({head_len}) must equal tail_block_size ({tail_block_size})"
                )
            }
            Self::BackendUnavailable => {
                write!(f, "requested SIMD backend is unavailable on this target")
            }
            Self::InvalidStateTransition => {
                write!(
                    f,
                    "cannot change hybrid tail-spreading mode mid-cycle; call before streaming or after reset()"
                )
            }
            Self::InvalidNonUniformConfig => {
                write!(
                    f,
                    "invalid non-uniform partitioned configuration (causality, contiguity, stage count, or budget violation)"
                )
            }
        }
    }
}

impl core::error::Error for ConvError {}
