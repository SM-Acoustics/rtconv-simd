# Validation & Benchmark Report (`VALIDATION.md`) — WO-RTCONV-002

## 1. Origin Commit & Acceptance Status

- **Baseline Origin Commit (WO-001)**: `fd79330` (`Baseline WO-001: rtconv_simd v0.1.0 (microfft 0.6.0 + fearless_simd 1.2.0)`)

| Gate | Status | Evidence Summary |
| :--- | :--- | :--- |
| **`FUNCTIONAL_PASS`** | **PASS** | 17/17 tests passed (`cargo test --release --features phastft`): all 3 engines + all 3 FFT backends (`MicroFftBackend`, `SimdFixedFftBackend` Scalar/SSE2/AVX2, `PhastFftBackend`) + `HybridConvolver` time-distributed tail spreading validated against independent `f64` DFT & convolution reference (round-trip `< 5e-6`, SNR `> 120 dB`, Parseval relative error `< 1e-5`). |
| **`RT_DESIGN_PASS`** | **PASS** | `#![no_std]` + `#![forbid(unsafe_code)]`; `cargo check --lib --no-default-features` exits `0`; `cargo tree --no-default-features -e normal` has zero `std`/`alloc`; custom `GlobalAlloc` tests confirm **0 allocations / 0 deallocations** in audio callbacks for `MicroFftBackend`, `SimdFixedFftBackend`, `PhastFftBackend`, and `HybridConvolver` (`tail_spreading = true`). |
| **`PERFORMANCE_PASS`** | **PASS** | `SimdFixedFftBackend` achieves **2.08x** (`N=64`, SSE2), **2.85x** (`N=128`, AVX2), and **2.51x** (`N=256`, AVX2) over `MicroFftBackend`. `Hybrid(H128/T128)` with `tail_spreading = true` + `SimdFixedFft(SSE2)` cuts peak p99 callback time at 16,383 taps from `30.10 µs` (`4.52%`) down to **`5.20 µs` (`0.78%` of the 666.67 µs deadline)** — a **5.79x peak p99 reduction** with 0 samples added latency. |
| **`TARGET_RT_VALIDATED`** | **NOT_VERIFIED** | Host is general-purpose Windows 11 Pro; observed maxima reflect OS scheduler preemption and are **not** formal WCET proofs. ARM64 NEON, WASM SIMD128, AVX-512, and RISC-V hardware targets were not executed. |

## 2. Host & Build Environment

- **Host**: `SalatielPC` — AMD Ryzen 3 PRO 2300U w/ Radeon Vega Mobile Gfx (4C/4T, x86-64 Zen 1; SSE2, SSE4.2, AVX2, FMA; no AVX-512)
- **OS**: Microsoft Windows 11 Pro, Version `10.0.26300` (64-bit)
- **Toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0 (3acc79c0c 2026-11-05)`
- **Profile**: `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`
- **`fearless_simd` Compatibility**: `rtconv_simd` uses [`fearless_simd 1.2.0`](https://docs.rs/fearless_simd/1.2.0); optional [`phastft 0.4.1`](https://docs.rs/phastft) depends on `fearless_simd 0.5.0`. Cargo resolves both semver-distinct versions cleanly under `--features phastft`, while `--no-default-features` compiles strictly `#![no_std]` with only `fearless_simd 1.2.0`.

## 3. Separated FFT Stage Benchmarks (`cargo bench --bench fft_backends_bench --features phastft`)

32 reps × 2,000 iterations (200 warmup), median ns/call:

| Backend | `N` | RFFT (ns) | IFFT (ns) | Total (ns) | Speedup vs MicroFFT | External Scratch |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MicroFftBackend` | 64 | 209.4 | 506.2 | 715.6 | 1.00x | 0 B (stack ~0.5 KiB) |
| `SimdFixedFft(Scalar)` | 64 | 209.4 | 281.2 | 490.6 | 1.46x | 256 B |
| **`SimdFixedFft(SSE2)`** | **64** | **153.1** | **190.6** | **343.8** | **2.08x** | **256 B** |
| `SimdFixedFft(AVX2)` | 64 | 218.8 | 143.8 | 362.5 | 1.97x | 256 B |
| `PhastFftBackend` | 64 | 228.1 | 268.8 | 496.9 | 1.44x | 256 B |
| `MicroFftBackend` | 128 | 443.8 | 1134.4 | 1578.1 | 1.00x | 0 B (stack ~1.0 KiB) |
| `SimdFixedFft(Scalar)` | 128 | 596.9 | 415.6 | 1012.5 | 1.56x | 512 B |
| `SimdFixedFft(SSE2)` | 128 | 415.6 | 368.8 | 784.4 | 2.01x | 512 B |
| **`SimdFixedFft(AVX2)`** | **128** | **300.0** | **253.1** | **553.1** | **2.85x** | **512 B** |
| `PhastFftBackend` | 128 | 268.8 | 287.5 | 556.2 | 2.84x | 512 B |
| `MicroFftBackend` | 256 | 884.4 | 2187.5 | 3071.9 | 1.00x | 0 B (stack ~2.1 KiB) |
| `SimdFixedFft(Scalar)` | 256 | 1115.6 | 821.9 | 1937.5 | 1.59x | 1024 B |
| `SimdFixedFft(SSE2)` | 256 | 709.4 | 578.1 | 1287.5 | 2.39x | 1024 B |
| **`SimdFixedFft(AVX2)`** | **256** | **700.0** | **525.0** | **1225.0** | **2.51x** | **1024 B** |
| `PhastFftBackend` | 256 | 603.1 | 762.5 | 1365.6 | 2.25x | 1024 B |

*Key finding on Ryzen 3 PRO 2300U (Zen 1)*: At `N = 64`, **SSE2 (`343.8 ns`) outperforms AVX2 (`362.5 ns`)** because Zen 1 double-pumps 256-bit AVX2 instructions over 128-bit execution pipes. At `N = 128` and `N = 256`, AVX2 takes the lead (`2.85x` and `2.51x` over `MicroFftBackend`), matching or beating `PhastFftBackend` while remaining 100% `#![no_std]` with zero `alloc`.

## 4. Full `PartitionedConvolver` FDL Comparison (48 kHz)

| FFT Backend | IR | Blk | Med (µs) | p99 (µs) | Max Obs. (µs) | p99 / Deadline % | Workspace (KiB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `MicroFftBackend` | 256 | 32 | 0.90 | 1.10 | 33.00 | 0.17% | 4.76 |
| `SimdFixedFft(SSE2)` | 256 | 32 | 0.60 | 0.60 | 0.60 | **0.09%** | 5.01 |
| `PhastFftBackend` | 256 | 32 | 1.20 | 1.40 | 304.80 | 0.21% | 5.01 |
| `MicroFftBackend` | 2048 | 128 | 4.10 | 6.10 | 60.10 | 0.23% | 34.76 |
| `SimdFixedFft(SSE2)` | 2048 | 128 | 2.20 | 2.20 | 37.90 | **0.08%** | 35.76 |
| `SimdFixedFft(AVX2)` | 2048 | 128 | 2.20 | 2.30 | 10.50 | 0.09% | 35.76 |
| `PhastFftBackend` | 2048 | 128 | 3.00 | 3.20 | 329.30 | 0.12% | 35.76 |
| `MicroFftBackend` | 16383 | 128 | 14.00 | 54.00 | 316.60 | 2.02% | 260.51 |
| `SimdFixedFft(SSE2)` | 16383 | 128 | 11.70 | 38.40 | 274.20 | 1.44% | 261.51 |
| `SimdFixedFft(AVX2)` | 16383 | 128 | 8.40 | 36.70 | 438.00 | 1.38% | 261.51 |
| `PhastFftBackend` | 16383 | 128 | 7.80 | 19.40 | 139.60 | **0.73%** | 261.51 |

## 5. Fase E — `Hybrid(H128/T128)` @ 32-Sample Host Blocks: 4-Phase Cycle (BURST vs SPREAD)

Deadline: `666.67 µs` (32 samples @ 48 kHz). Latency: **0 samples** in both modes.

| FFT Backend | IR | Mode | Med P0..P3 (µs) | p99 P0..P3 (µs) | Worst p99 / 666.67 µs |
| :--- | ---: | :--- | :--- | :--- | ---: |
| `MicroFftBackend` | 2048 | BURST | `1.00 / 1.00 / 1.00 / 4.70` | `1.10 / 1.10 / 1.10 / 5.70` | 0.85% |
| `MicroFftBackend` | 2048 | SPREAD | `1.90 / 2.00 / 2.00 / 6.00` | `3.30 / 3.30 / 3.40 / 15.60` | 2.34% *(no gain at 2048)* |
| `SimdFixedFft(SSE2)` | 2048 | BURST | `1.30 / 1.30 / 1.30 / 4.00` | `1.90 / 1.90 / 1.80 / 6.30` | 0.95% |
| **`SimdFixedFft(SSE2)`** | **2048** | **SPREAD** | **`1.30 / 1.30 / 1.30 / 2.40`** | **`1.40 / 1.40 / 1.40 / 2.80`** | **0.42%** |
| `MicroFftBackend` | 16383 | BURST | `1.30 / 1.30 / 1.30 / 13.40` | `3.00 / 2.90 / 2.90 / 30.10` | 4.52% |
| `MicroFftBackend` | 16383 | SPREAD | `4.70 / 4.70 / 4.70 / 5.80` | `6.90 / 7.00 / 7.10 / 8.70` | 1.30% (3.46x peak cut) |
| `SimdFixedFft(SSE2)` | 16383 | BURST | `1.00 / 1.00 / 1.00 / 8.80` | `1.70 / 1.60 / 1.70 / 16.90` | 2.54% |
| **`SimdFixedFft(SSE2)`** | **16383** | **SPREAD** | **`3.20 / 3.20 / 3.20 / 2.50`** | **`5.20 / 5.10 / 4.90 / 4.00`** | **0.78% (3.25x peak cut)** |
| `SimdFixedFft(AVX2)` | 16383 | BURST | `1.20 / 1.20 / 1.20 / 10.20` | `2.70 / 2.70 / 2.70 / 24.80` | 3.72% |
| `SimdFixedFft(AVX2)` | 16383 | SPREAD | `3.20 / 3.20 / 3.20 / 2.60` | `5.20 / 5.00 / 5.20 / 4.20` | **0.78% (4.77x peak cut)** |
| `PhastFftBackend` | 16383 | BURST | `1.10 / 1.10 / 1.10 / 9.00` | `2.20 / 1.90 / 2.00 / 59.60` | 8.94% |
| `PhastFftBackend` | 16383 | SPREAD | `3.40 / 3.40 / 3.40 / 2.50` | `6.60 / 5.90 / 8.60 / 4.70` | 1.29% (6.93x peak cut) |

## 6. Disassembly SIMD Evidence & Binary Size

- **Assembly inspection** (`cargo rustc --lib --release -- --emit=asm` on `rtconv_simd`):
  - WO-001 baseline (`MicroFftBackend` only): `FMA = 0`, `YMM = 0`, `XMM = 1,584`.
  - WO-RTCONV-002 (with `SimdFixedFftBackend`): **`FMA = 96` (`vfmadd`/`vfmsub`/`vfnmadd`), `YMM = 527`, `XMM = 5,251`** emitted directly in `librtconv_simd`, proving genuine SIMD vectorization inside the FFT butterfly transforms themselves (not only in CMAC).
- **Library artifact size**:
  - `librtconv_simd.rlib` (`--no-default-features`): `754,634 B` (`736.9 KiB`)
  - `librtconv_simd.rlib` (`--features phastft`): `780,838 B` (`762.5 KiB`, `+26.2 KiB` wrapper delta)
- **Default backend recommendation**: Keep `MicroFftBackend` as the default generic type parameter (`PartitionedConvolver<'a, F = MicroFftBackend>`) for backwards compatibility, and recommend `SimdFixedFftBackend` (with `SSE2` at `N=64` or `AVX2` at `N=128/256`, plus `tail_spreading = true` for IRs ≥ 2,048 taps) for maximum real-time margin on x86-64.
