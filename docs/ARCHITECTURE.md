# Architecture (`docs/ARCHITECTURE.md`)

## Layering & Independence

In the SM-Acoustics layered model (`SM-ENG-001`, *Proposed v1*), `rtconv-simd` sits at **L1 (Portable DSP Algorithms)**:

- **Zero platform coupling**: No dependency on ESP-IDF, FreeRTOS, RTOS, HAL, OS audio drivers, MIDI, USB, networking, file I/O, or GUI frameworks.
- **Core invariant**: `#![no_std]` and `#![forbid(unsafe_code)]`, with `default = []` (no `std` or `alloc`).
- **Phase-agnostic semantics**: Any finite real causal FIR impulse response `&[f32]` is processed verbatim without phase classification, conversion, or normalization.

## Module Structure

- `src/lib.rs`: Public API, `SimdBackend`, `Capabilities`, `EngineKind`.
- `src/error.rs`: Typed `ConvError` implementing `core::error::Error`.
- `src/simd_kernels.rs`: Scalar and `fearless_simd 1.2.0` dot-product and split-complex (SoA) CMAC kernels (including partition sub-range accumulation for time-distributed FDL).
- `src/fft.rs`: `FftBackend` trait, `FftSize` (`N64`, `N128`, `N256`), and `MicroFftBackend` (`microfft 0.6.0`).
- `src/simd_fft.rs` + `src/simd_fft_tables.rs`: `#![no_std]` `SimdFixedFftBackend` using `fearless_simd 1.2.0`, `N/2`-point CFFT for R2C/C2R, radix-4 stages 0–1, SIMD butterflies for `half >= 4`, and precomputed `const` twiddles.
- `src/phastft_backend.rs` (`#[cfg(feature = "phastft")]`): Desktop SIMD `PhastFftBackend` wrapping `phastft 0.4.1` (`default-features = false`).
- `src/direct.rs`: `DirectConvolver` (reverse-indexed circular buffer, unit-stride forward SIMD dot product).
- `src/partitioned.rs`: `PartitionedConvolver` (50% overlap-save uniform FDL, single-shot and time-distributed helpers).
- `src/hybrid.rs`: `HybridConvolver` (direct FIR head + partitioned FFT tail with `head_len == tail_block_size`, 0-sample latency, optional `set_tail_spreading(true)`).
