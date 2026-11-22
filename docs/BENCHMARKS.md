# Benchmarks (`docs/BENCHMARKS.md`)

**Provenance**: `PROVEN` (`HOST_ONLY`). Empirical measurements on general-purpose Windows 11 Pro — **not** formal WCET proofs.

## Host Environment

- **CPU**: AMD Ryzen 3 PRO 2300U w/ Radeon Vega Mobile Gfx (4C/4T, x86-64 Zen 1; SSE2, SSE4.2, AVX2, FMA)
- **OS**: Microsoft Windows 11 Pro `10.0.26300` (64-bit)
- **Toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, host `x86_64-pc-windows-gnu` (`rustup` default `stable-x86_64-pc-windows-gnu`), `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`
- **Nominal Audio Budget**: `48,000 Hz`, `32` samples/block → **`666.67 µs` deadline**
- **Commands**:
  - `cargo bench --bench rtconv_bench --all-features`
  - `cargo bench --bench fft_backends_bench --all-features`

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

## 3. Isolated FIR Dot Product & CMAC Notes

- `dot_product`: AVX2+FMA reaches **8.83x** over scalar at 1,024 taps (`882.8 ns → 100.0 ns`); at 32–64 taps on Zen 1, SSE2 (`7.8–10.9 ns`) outperforms AVX2 (`10.9–14.1 ns`).
- `cmac_fdl`: Because LLVM auto-vectorizes `cmac_scalar` into SSE2 at `opt-level = 3`, explicit AVX2 beats scalar at `N=128` (`1.53x`) but shows slight overhead at `N=64` (`0.73x`) and `N=256` (`0.81x`) on Zen 1.
