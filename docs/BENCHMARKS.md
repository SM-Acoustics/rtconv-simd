# Benchmarks (`docs/BENCHMARKS.md`)

**Provenance**: `PROVEN` (`HOST_ONLY`). Empirical measurements on general-purpose Windows 11 Pro — **not** formal WCET proofs.

## Host Environment

- **CPU**: AMD Ryzen 3 PRO 2300U w/ Radeon Vega Mobile Gfx (4C/4T, x86-64 Zen 1; SSE2, SSE4.2, AVX2, FMA)
- **OS**: Microsoft Windows 11 Pro `10.0.26300` (64-bit)
- **Toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, host `x86_64-pc-windows-gnu` (`rustup` default `stable-x86_64-pc-windows-gnu`), `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`
- **Nominal Audio Budget**: `48,000 Hz`, `32` samples/block -> **`666.67 µs` deadline**
- **Commands**:
  - `cargo bench --bench rtconv_bench --all-features`
  - `cargo bench --bench fft_backends_bench --all-features`
  - `cargo bench --bench nonuniform_bench --features nonuniform`
  - `cargo bench --bench fixed_q15_bench --features fixed-q15`

## 1. Isolated FFT Transforms (`RFFT + IFFT`, ns/call)

| Backend | `N = 64` (Total / vs Micro) | `N = 128` (Total / vs Micro) | `N = 256` (Total / vs Micro) |
| :--- | ---: | ---: | ---: |
| `MicroFftBackend` | 715.6 ns (1.00x) | 1,578.1 ns (1.00x) | 3,071.9 ns (1.00x) |
| `SimdFixedFft(Scalar)` | 490.6 ns (1.46x) | 1,012.5 ns (1.56x) | 1,937.5 ns (1.59x) |
| `SimdFixedFft(SSE2)` | **343.8 ns (2.08x)** | 784.4 ns (2.01x) | 1,287.5 ns (2.39x) |
| `SimdFixedFft(AVX2)` | 362.5 ns (1.97x) | **553.1 ns (2.85x)** | **1,225.0 ns (2.51x)** |
| `PhastFftBackend` | 496.9 ns (1.44x) | 556.2 ns (2.84x) | 1,365.6 ns (2.25x) |

## 2. `Hybrid(H128/T128)` @ 32-Sample Host Blocks (4-Phase Cycle: BURST vs SPREAD)

| FFT Backend | IR Taps | Mode | Med P0..P3 (µs) | p99 P0..P3 (µs) | Worst p99 / 666.67 µs |
| :--- | ---: | :--- | :--- | :--- | ---: |
| `MicroFftBackend` | 16,383 | BURST | `1.30 / 1.30 / 1.30 / 13.40` | `3.00 / 2.90 / 2.90 / 30.10` | 4.52% |
| `MicroFftBackend` | 16,383 | SPREAD | `4.70 / 4.70 / 4.70 / 5.80` | `6.90 / 7.00 / 7.10 / 8.70` | 1.30% |
| `SimdFixedFft(SSE2)` | 16,383 | BURST | `1.00 / 1.00 / 1.00 / 8.80` | `1.70 / 1.60 / 1.70 / 16.90` | 2.54% |
| **`SimdFixedFft(SSE2)`** | **16,383** | **SPREAD** | **`3.20 / 3.20 / 3.20 / 2.50`** | **`5.20 / 5.10 / 4.90 / 4.00`** | **0.78%** |
| `SimdFixedFft(AVX2)` | 16,383 | SPREAD | `3.20 / 3.20 / 3.20 / 2.60` | `5.20 / 5.00 / 5.20 / 4.20` | **0.78%** |
| `PhastFftBackend` | 16,383 | SPREAD | `3.40 / 3.40 / 3.40 / 2.50` | `6.60 / 5.90 / 8.60 / 4.70` | 1.29% |

## 3. Multi-Stage NUPC vs `Hybrid` (`f32`, `SimdFixedFft(SSE2)`, 32-Sample Blocks)

| Engine | IR Taps | Mean (µs) | p95 (µs) | p99 (µs) | p99 / 666.67 µs | Workspace |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Hybrid(H128/T128) BURST` | 2,048 | 1.71 | 3.70 | 3.70 | 0.56% | 36,600 B |
| `Hybrid(H128/T128) SPREAD` | 2,048 | 2.55 | 3.80 | 4.70 | 0.71% | 36,600 B |
| `NonUniform(D32/32/64/128)` | 2,048 | 6.27 | 10.00 | 10.50 | 1.57% | 44,744 B |
| `Hybrid(H128/T128) BURST` | 16,383 | 5.70 | 18.70 | 20.20 | 3.03% | 267,768 B |
| `Hybrid(H128/T128) SPREAD` | 16,383 | 5.92 | 6.00 | 8.30 | 1.25% | 267,768 B |
| `NonUniform(D32/32/64/128)` | 16,383 | 8.91 | 13.60 | 20.50 | 3.08% | 275,912 B |

**Honest Interpretation**: While the core FFT ceiling remains `N = 256` (`B = 128`), `Hybrid(H128/T128) SPREAD` stays faster than `NonUniform(D32/32/64/128)` on x86-64 because both share the same tail partition size (`B = 128`) while NUPC runs two additional intermediate stages (`B = 32`, `B = 64`). Unlocking NUPC throughput wins on long IRs requires `N512`/`N1024+` FFT sizes (tracked in `docs/ROADMAP.md`).

## 4. Fixed-Point Q1.15 Direct FIR vs `f32` Direct (32-Sample Blocks, x86-64 with HW FPU)

| Taps | Q15 Direct (µs / % budget) | Q15 RAM | `f32` Scalar (µs) | `f32` SIMD (µs) | `f32` RAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.35 µs (0.20%) | 128 B | 0.61 µs | 0.65 µs | 256 B |
| 64 | 2.53 µs (0.38%) | 256 B | 1.17 µs | 0.77 µs | 512 B |
| 128 | 5.83 µs (0.87%) | 512 B | 3.77 µs | 1.14 µs | 1,024 B |
| 256 | 9.80 µs (1.47%) | 1,024 B | 8.43 µs | 2.15 µs | 2,048 B |
| 512 | 19.34 µs (2.90%) | 2,048 B | 17.45 µs | 4.21 µs | 4,096 B |

*Note*: On x86-64 with hardware FPU/SSE2, `f32` SIMD wins in throughput while Q1.15 cuts workspace RAM in half (`50%`). Q1.15 compute speedups target FPU-less microcontrollers (e.g., RV32IMAC ESP32-C6), where software `f32` emulation is expensive.

## 5. Isolated FIR Dot Product & CMAC Notes

- `dot_product`: AVX2+FMA reaches **8.83x** over scalar at 1,024 taps (`882.8 ns -> 100.0 ns`); at 32–64 taps on Zen 1, SSE2 (`7.8–10.9 ns`) outperforms AVX2 (`10.9–14.1 ns`).
- `cmac_fdl`: Because LLVM auto-vectorizes `cmac_scalar` into SSE2 at `opt-level = 3`, explicit AVX2 beats scalar at `N=128` (`1.53x`) but shows slight overhead at `N=64` (`0.73x`) and `N=256` (`0.81x`) on Zen 1.
