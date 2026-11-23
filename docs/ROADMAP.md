# Roadmap (`docs/ROADMAP.md`)

Evolution plan for `rtconv-simd` beyond `v0.1.0`. Status labels strictly separate what is implemented on `main` from what remains open or requires physical hardware verification.

## Status Overview

| # | Initiative | Status | Current Reality on `main` |
| :---: | :--- | :--- | :--- |
| **1** | **Non-Uniform Partitioned Convolution (NUPC / Gardner)** | **Implemented (`B <= 128`)** · *Large FFTs Pending* | Multi-stage `NonUniformConvolver` (`f32`, `--features nonuniform`) and `FixedQ15NonUniformConvolver` (`Q1.15/BFP`, `--features fixed-nonuniform`) implemented up to 8 stages with zero-latency causality (`D_s >= B_s`), sample-time `NonUniformScheduler`, `SPREAD`, and `NonUniformBudget`. **Pending**: `N512` / `N1024` / `N2048` FFT sizes so NUPC beats `Hybrid SPREAD` on long IRs (`>= 16k` taps). |
| **2** | **FDL RAM Reduction (Compact / Block Floating Point)** | **Implemented (Q1.15/BFP)** · *`f16` / Hybrid Pending* | Full Q1.15 + Block Floating Point stack (`fixed-q15`, `fixed-fft`, `fixed-nonuniform`) cuts workspace RAM by **~50%** vs `f32` (`512 B` twiddle ROM). **Pending**: optional `f16` or mixed `f32`-head / BFP-tail storage for `f32` pipelines. |
| **3** | **Hardware-in-the-Loop Qualification (`DEVICE_VERIFIED`)** | **In Progress (`CROSS_COMPILE_VERIFIED`)** | Standalone bare-metal ESP32-C6 harness (`hardware/esp32c6/`, `riscv32imac-unknown-none-elf`) compiles cleanly (`CROSS_COMPILE_VERIFIED = PASS`). **Pending**: authorized physical flashing + `mcycle` CSR profiling on ESP32-C6, Cortex-M7, and ARM64 NEON (`DEVICE_VERIFIED = NO`). |
| **4** | **Immutable Shareable IR Spectra (`&[f32]` / `&[i16]` ROM)** | **Planned** | Partition spectra are already isolated in split `ir_re`/`ir_im` (+ `ir_exp` in BFP) layouts, ready for a read-only Flash/ROM-backed spectrum type shared across stereo/multichannel instances (nearly 50% RAM savings for fixed speaker/cabinet IRs). |
| **5** | **Multi-Channel & True-Stereo Helpers** | **Planned** | Core engines remain strictly mono; zero-allocation stereo, multi-channel, and true-stereo (4-convolver matrix) ergonomic wrappers are planned on top of independent instances. |
| **6** | **Glitch-Free IR Crossfade** | **Planned** | Pre-allocated dual-engine crossfade helper for live IR switching without heap allocations or clicks. |

---

## Release Milestones

### `0.1.0` — Released on GitHub (`v0.1.0`, `crates.io` pending)
- `#![no_std]`, `#![forbid(unsafe_code)]`, caller-owned `&mut [f32]` workspaces.
- `DirectConvolver`, `PartitionedConvolver` (`N64/N128/N256`), and zero-latency `HybridConvolver` with `SPREAD` tail distribution.
- `MicroFftBackend`, `SimdFixedFftBackend` (`fearless_simd 1.2.0`), and optional `PhastFftBackend`.

### `0.2.0` — Implemented on `main` (Pre-Release)
- **CI & Zero-Alloc Hardening (`WO-RTCONV-004A`)**: Standalone `harness = false` `tests/no_alloc_rt.rs` with positive allocation control (eliminating macOS `libtest` flake) and SHA-pinned MSRV `1.89.0` workflows.
- **Multi-Stage NUPC `f32` (`WO-RTCONV-004B`)**: `NonUniformPlan`, `NonUniformScheduler`, `NonUniformBudget`, `NonUniformConvolver` (`--features nonuniform`).
- **Fixed-Point Q1.15 / BFP Stack (`WO-RTCONV-004C/D/E`)**: `FixedQ15DirectConvolver`, `rfft_q15_bfp`/`irfft_q15_bfp`, `FixedQ15PartitionedConvolver`, and `FixedQ15NonUniformConvolver` (`--features fixed-q15`, `fixed-fft`, `fixed-nonuniform`).
- **ESP32-C6 Bare-Metal Harness (`WO-RTCONV-004F`)**: `hardware/esp32c6/` (`CROSS_COMPILE_VERIFIED`).

### Next Priorities (`0.2.x` / `0.3.0`)
1. **Large FFT Sizes (`N512`, `N1024`, `N2048`)** across `FftBackend` (`SimdFixedFftBackend`, `MicroFftBackend`, `PhastFftBackend`, and `fft_q15`) to unlock NUPC throughput advantages over `Hybrid SPREAD` on long room IRs (`16k–131k` taps).
2. **Physical ESP32-C6 Profiling (`DEVICE_VERIFIED`)**: Authorized flashing and `mcycle` CSR measurements recorded under `hardware/esp32c6/measurements/`.
3. **Immutable Shared IR Spectra (`Flash/ROM`)** for `f32` and `Q1.15/BFP`.
4. **Stereo / True-Stereo Wrappers & Glitch-Free Dual-Engine Crossfade**.
5. **`crates.io` Publication** (publish `0.1.0` or cut `0.2.0` directly once `0.2.0` gates close).
