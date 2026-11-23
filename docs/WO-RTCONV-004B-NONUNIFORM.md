# WO-RTCONV-004B — Non-Uniform Partitioned Convolution (NUPC)

**Work Order**: `RTCONV-004B`  
**Branch**: `feat/rtconv-004b-nonuniform`  
**Feature Gate**: `nonuniform` (`default = []` preserved; `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap)  
**Verification Scope**: `HOST_TESTED` (`PROVEN` on Windows x86-64 `rustc 1.100.0` + MSRV `1.89.0`; GitHub Actions CI covers Linux/macOS/Windows/MSRV).

---

## 1. Mathematical & Temporal Contract

`NonUniformConvolver` decomposes a causal FIR impulse response $h[n]$ into up to `MAX_NONUNIFORM_STAGES = 8` contiguous, non-overlapping regions:

$$h[n] = \sum_{s=0}^{S-1} h_s[n - D_s], \qquad y[n] = \sum_{s=0}^{S-1} (x * h_s)[n - D_s]$$

- **Stage 0 (Direct FIR)**: `ir_offset == 0`, `block_size == 1`, zero algorithmic latency.
- **Stages $s \ge 1$ (FDL)**: Uniform 50% overlap-save partitions with `block_size` $B_s \in \{32, 64, 128\}$ (FFT sizes `N64`, `N128`, `N256` using the existing `FftBackend` — no new external FFT dependency). `N512`/`N1024` are explicitly **not** claimed supported yet.
- **Strict Zero-Latency Causality**: `NonUniformPlan::validate()` enforces $D_s \ge B_s$ for every FDL stage and rejects non-causal, gapped, or overlapping plans with `ConvError::InvalidNonUniformConfig`.
- **Sample-Time Scheduler & SPREAD**: `NonUniformScheduler` operates strictly in sample time (shared verbatim with `RTCONV-004E`). Each FDL stage pre-seeds its causal output delay ring (`ir_offset + block_size` samples) with `ir_offset` zeros so output sample $y_s[0]$ aligns with global sample $t = D_s$. When `slack_samples = D_s - B_s > 0`, CMAC partitions spread across `min(slack_samples, B_s - 1)` samples so consecutive blocks never collide and `deadline_misses()` stays `0`.
- **Admission Control**: `NonUniformBudget` + `NonUniformPlan::check_budget()` / `prepare_with_budget()` reject configurations exceeding declared memory or operation limits with `BudgetVerdict::Exceeded` -> `ConvError::InvalidNonUniformConfig`, and return `BudgetVerdict::Unverified` when no bounds are supplied (never claiming formal WCET).

---

## 2. Numerical & Real-Time Verification

- **Reference Parity (`tests/nonuniform_reference.rs`)**: Impulses at every stage boundary (`0, 1, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 2047`), non-multiple IR lengths (`1..2051`), both `MicroFftBackend` and `SimdFixedFftBackend`, multitone + silence tail drain, bit-identical `reset()`, and negative tests (non-causal & budget rejection): **all PASS** (`max_abs < 1e-4`, `SNR > 95 dB` vs `f64`, `deadline_misses == 0`).
- **Fragmentation & Isolation (`tests/nonuniform_boundaries.rs`)**: Dynamic host block sizes `[1, 7, 13, 32, 3, 64, 19, 100, 17, 5, 31]` and independent stereo instances: **PASS**.
- **Zero Heap Allocations (`tests/no_alloc_rt.rs`)**: Standalone `harness = false` binary with positive control verifies `0` allocations / `0` deallocations in `nonuniform::{prepare, process_block, reset, metadata}` on both `rustc 1.100.0` and MSRV `1.89.0`: **PASS**.

---

## 3. Empirical Benchmark Comparison (`HOST_ONLY`)

**Host**: AMD Ryzen 3 PRO 2300U (x86-64 Zen 1, `fearless_simd::Sse2`), Windows 11 Pro `10.0.26300`, `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `x86_64-pc-windows-gnu`, `opt-level = 3`, `lto = "thin"`, `48 kHz`, `block = 32` (`666.67 µs` budget), `cargo bench --bench nonuniform_bench --features nonuniform`.

| Engine | IR Taps | Mean (µs) | p95 (µs) | p99 (µs) | p99 / Budget | Max (µs) | Workspace |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Hybrid(H128/T128) BURST` | 2,048 | 1.71 | 3.70 | 3.70 | 0.56% | 13.20 | 36,600 B |
| `Hybrid(H128/T128) SPREAD` | 2,048 | 2.55 | 3.80 | 4.70 | 0.71% | 109.50 | 36,600 B |
| `NonUniform(D32/32/64/128)` | 2,048 | 6.27 | 10.00 | 10.50 | 1.57% | 151.00 | 44,744 B |
| `Hybrid(H128/T128) BURST` | 16,383 | 5.70 | 18.70 | 20.20 | 3.03% | 91.70 | 267,768 B |
| `Hybrid(H128/T128) SPREAD` | 16,383 | 5.92 | 6.00 | 8.30 | 1.25% | 775.00 | 267,768 B |
| `NonUniform(D32/32/64/128)` | 16,383 | 8.91 | 13.60 | 20.50 | 3.08% | 118.90 | 275,912 B |

**Honest Interpretation**: With the current core FFT ceiling of `N256` (`B = 128`), `Hybrid(H128/T128) SPREAD` remains faster than `NonUniform(D32/32/64/128)` on x86-64 SSE2 because both share the same tail partition size (`B = 128`) while NUPC runs two extra intermediate FDL stages (`B = 32`, `B = 64`). NUPC throughput wins require larger tail FFTs (`N512`/`N1024`) in a follow-up stage; today NUPC delivers the validated sample-time multi-stage scheduler, zero-latency causal planning, budget admission control, and reusable `Planning Plane` for `RTCONV-004E`.
