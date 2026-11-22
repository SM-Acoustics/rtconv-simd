# `rtconv_simd` — Hard Real-Time, `#![no_std]` FIR Convolution Engine

`rtconv_simd` is an independent, `#![no_std]`, `#![forbid(unsafe_code)]`, allocation-free Rust crate implementing three real-time `f32` FIR convolution engines with [`fearless_simd 1.2.0`](https://docs.rs/fearless_simd/1.2.0) vectorization and three pluggable `FftBackend` implementations.

## Architecture & Engines

1. **`DirectConvolver`**: Time-domain FIR convolution (`y[n] = Σ h[k]·x[n-k]`) with reverse-indexed circular history buffer (unit-stride forward SIMD loads, zero per-sample copies). Latency: **0 samples**.
2. **`PartitionedConvolver`**: Uniform Frequency-Domain Delay Line (FDL) using **50% Overlap-Save (OLS)** for `block_size ∈ {32, 64, 128}` (`FFT ∈ {64, 128, 256}`). Spectra are stored in split-complex (SoA) planes (`N/2 + 1` bins, `im[0] = im[N/2] = 0.0`) for contiguous SIMD CMAC. Supports both single-shot `process_block()` and time-distributed FDL helpers (`reset_accumulator`, `accumulate_past_partitions`, `finish_distributed_block`).
3. **`HybridConvolver`**: Zero-latency combination of a direct FIR head (`h[0..head_len]`) and a partitioned FFT tail (`h[head_len..]`, `head_len == tail_block_size`). Supports optional **zero-latency time-distributed tail spreading** (`set_tail_spreading(true)`), which spreads past partition CMACs (`p = 1..P-1`) across pre-completion host callbacks (e.g., phases 0, 1, 2 of a 4-callback `Hybrid(H128/T128)` @ 32-sample cycle) without altering temporal alignment or adding latency.

## FFT Backends (`FftBackend`)

| Backend | Target / Feature | SIMD FFT | External Scratch (`N = 64/128/256`) | Peak Stack | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`MicroFftBackend`** | `#![no_std]` (default) | No (scalar radix-2) | `0 B` | ~2.1 KiB (`N=256`) | Baseline using [`microfft 0.6.0`](https://docs.rs/microfft/0.6.0). |
| **`SimdFixedFftBackend`** | `#![no_std]` (always available) | **Yes** (`fearless_simd 1.2.0`) | `256 / 512 / 1024 B` | `< 256 B` | Radix-4 stages 0–1 + SIMD butterflies (`half >= 4`), `N/2`-point CFFT for both RFFT and C2R, precomputed `const` twiddles, explicit Scalar/SSE2/AVX2 selection. **Fastest overall on Ryzen 3 PRO 2300U (2.08x–2.85x over `microfft`).** |
| **`PhastFftBackend`** | Desktop (`--features phastft`) | **Yes** (`phastft 0.4.1`) | `256 / 512 / 1024 B` | `< 256 B` | Wraps [`phastft 0.4.1`](https://docs.rs/phastft) (`default-features = false`, no `rayon`) using `r2c_fft_f32_with_planner_and_opts` and `c2r_fft_f32_with_planner_and_opts` with pre-initialized `PlannerR2c32` and zero callback allocations. |

## Feature Flags

- `default = ["std"]`: Enables `fearless_simd/std` for one-time runtime CPUID detection via `SimdBackend::detect()`. The DSP core never uses `std` or `alloc`.
- `--no-default-features`: Pure `#![no_std]` build (`fearless_simd` with `libm` + `force_support_fallback`, `microfft 0.6.0`, `SimdFixedFftBackend`). Zero `std`, zero `alloc`.
- `phastft`: Enables desktop `PhastFftBackend` (`phastft 0.4.1`, `default-features = false`).

## Verification & Reports

- [`VALIDATION.md`](./VALIDATION.md): Baseline WO-001 vs WO-RTCONV-002 comparison, separated RFFT/IFFT benchmarks, 4-phase Hybrid burst vs spread measurements, and ASM SIMD disassembly evidence.
- [`RT_AUDIT.md`](./RT_AUDIT.md): Real-time invariants, zero-alloc callback proofs across all three FFT backends, stack/panic audit, and platform limitations.

## License

Dual-licensed under `MIT OR Apache-2.0` (`LICENSE-MIT`, `LICENSE-APACHE`).
