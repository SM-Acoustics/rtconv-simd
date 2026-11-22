# Changelog

All notable changes to `rtconv-simd` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased (Release Candidate)

### Added
- Pure Rust `#![no_std]`, `#![forbid(unsafe_code)]` FIR convolution core (`rtconv_simd`) with zero heap allocation in `prepare` and audio processing.
- Three convolution engines: `DirectConvolver`, `PartitionedConvolver` (uniform 50% overlap-save FDL), and zero-latency `HybridConvolver`.
- Time-distributed FDL tail spreading in `HybridConvolver` (`prepare_with_tail_spreading`, `set_tail_spreading`) with `ConvError::InvalidStateTransition` protection against mid-cycle mode changes.
- Pluggable `FftBackend` trait with `MicroFftBackend` (`microfft 0.6.0`), `SimdFixedFftBackend` (`fearless_simd 1.2.0`, `no_std`, `N = 64, 128, 256`), and optional desktop `PhastFftBackend` (`phastft 0.4.1`, `--features phastft`).
- Explicit `Capabilities::simd_active` and `SimdBackend::is_simd_active()` distinguishing active vector SIMD from scalar fallback.
- Comprehensive validation suite against an independent `f64` reference, serialized zero-allocation `GlobalAlloc` tests, separated and integrated benchmarks, examples, and GitHub Actions CI workflows.
