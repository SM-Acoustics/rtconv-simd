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

## 6. IEEE 754 Subnormal Robustness (`cargo bench --bench denormal_bench`)

Linux x86-64 container (`fearless_simd::Sse2`), 1024 taps, 32-sample blocks, µs per block, `HOST_ONLY`. "Before" is `main` at `bdb3589`; "after" adds `DENORMAL_GUARD`.

| Case | Direct before | Direct after | Hybrid before | Hybrid after |
| :--- | ---: | ---: | ---: | ---: |
| Normal IR, input 0.1 | 4.57 | 4.29 | 0.93 | 1.01 |
| Normal IR, input 1e-35 (subnormal products) | 142.75 | 4.11 | 1.57 | 1.02 |
| Normal IR, input 1e-37 (subnormal products) | 346.30 | 4.33 | 48.08 | 0.92 |
| Normal IR, input 1e-39 (subnormal input) | 342.04 | 4.15 | 47.66 | 0.96 |
| Deep-tail IR (subnormal `f32` taps), input 0.1 | 67.37 | 4.15 | 1.07 | 0.92 |

---

## 7. Large FFTs (`N512`–`N16384`) & 16k / 64k Taps NUPC vs Hybrid — PR-A (`#14`, `HOST_ONLY`)

**Provenance**: `PROVEN` (`HOST_ONLY`). Empirical measurements on a shared virtualized Linux host — **not** formal WCET proofs (`DEVICE_VERIFIED = NO`).

### Host Environment & Metrology Hygiene (GUM ISO/IEC Guide 98-3 & Wefers Ch. 7)

- **CPU**: 2 vCPU `Intel(R) Xeon(R) CPU @ 2.30GHz` (`x86_64`, SSE2, SSE4.2, AVX2, FMA)
- **OS**: Linux `6.18.57` (`Ubuntu 24.04` VM)
- **Toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `x86_64-unknown-linux-gnu`, `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`
- **Nominal Audio Budget**: `48,000 Hz`, `32` samples/block → **`666.67 µs` deadline**
- **GUM Type A & Outlier Policy**: Each series discards an explicit warm-up transient (`200` / `64` calls in `fft_backends_bench`, `512` blocks in `nonuniform_bench`; Wefers §7.3.4) and retains **all** $n$ steady-state samples (`0` outliers trimmed) so tail percentiles (`p50`, `p95`, `p99`, `max`) honestly reflect both periodic monolithic FFT bursts and OS scheduling jitter. Expanded uncertainty is reported at $k = 2$ ($\approx 95\%$ coverage, $U = 2 u_A = 2 s / \sqrt{n}$), rounded half-to-even to 2 significant digits of $U$ with the mean aligned to the same decimal place (**ISO 80000-1 Annex B / ABNT NBR 5891**).
- **Commands**:
  - `cargo +1.100.0 bench --bench fft_backends_bench --all-features`
  - `cargo +1.100.0 bench --bench nonuniform_bench --features nonuniform` (SSE2 baseline)
  - `cargo +1.100.0 bench --bench nonuniform_bench --features "nonuniform std"` (AVX2+FMA)
  - `cargo +1.100.0 test --release --all-features --test aes17_audio_metrics -- --nocapture`

### 7.1 Isolated FFT Transforms `N = 64 ..= 16384` (`RFFT + IFFT`, ns/call)

| `N` | `n` | `MicroFft` med (`mean ± U`) | `SimdFixed(Scalar)` med (`mean ± U`) | `SimdFixed(SSE2)` med (`mean ± U`) | `SimdFixed(AVX2)` med (`mean ± U`) | `PhastFft` med (`mean ± U`) | Scratch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2000 | 590.5 (`785 ± 17`) | 984.3 (`1235 ± 21`) | 609.7 (`735 ± 16`) | 635.0 (`874 ± 24`) | 736.7 (`942 ± 22`) | 256 B |
| 128 | 2000 | 1,253.5 (`1333 ± 15`) | 2,325.5 (`2454 ± 20`) | 1,272.0 (`1359 ± 16`) | **1,234.0** (`1850 ± 210`) | **573.0** (`704 ± 15`) | 512 B |
| 256 | 2000 | 2,625.8 (`2898 ± 67`) | 4,887.8 (`5231 ± 32`) | 2,656.7 (`3062 ± 38`) | **2,492.0** (`2710 ± 33`) | **1,029.5** (`1290 ± 23`) | 1,024 B |
| 512 | 2000 | 5,393.6 (`5954 ± 55`) | 10,976.8 (`12010 ± 130`) | 5,532.7 (`5926 ± 37`) | **5,101.8** (`5691 ± 51`) | **1,992.7** (`2348 ± 29`) | 2,048 B |
| 1024 | 2000 | 11,591.6 (`13240 ± 150`) | 24,121.8 (`26650 ± 230`) | **11,476.4** (`12660 ± 540`) | **10,432.5** (`11149 ± 94`) | **3,790.1** (`4162 ± 99`) | 4,096 B |
| 2048 | 2000 | 24,964.9 (`26390 ± 120`) | 53,492.9 (`56440 ± 370`) | **23,851.0** (`25300 ± 140`) | **21,454.9** (`22960 ± 170`) | **7,869.8** (`8371 ± 76`) | 8,192 B |
| 4096 | 512 | 54,161.5 (`69500 ± 2900`) | 114,414.5 (`133900 ± 5600`) | **49,640.5** (`54300 ± 2100`) | **43,819.5** (`47700 ± 1700`) | **16,816.0** (`18180 ± 610`) | 16,384 B |
| 8192 | 512 | 118,206.0 (`130800 ± 2800`) | 248,908.5 (`270900 ± 3700`) | **106,512.0** (`125600 ± 2900`) | **92,539.0** (`101500 ± 1700`) | **36,360.0** (`38850 ± 730`) | 32,768 B |
| 16384 | 512 | 245,519.0 (`257200 ± 1600`) | 561,526.0 (`595900 ± 9400`) | 319,409.0 (`351100 ± 5000`) | 248,884.0 (`282200 ± 7000`) | **77,178.5** (`92000 ± 2800`) | 65,536 B |

*Notes*:
- `MicroFftBackend` gained ~2× on `rfft_inverse` after switching from a stack-allocated `N`-point complex IFFT to the zero-stack Hermitian real-FFT identity, while preserving `< 256 B` stack across all sizes.
- `SimdFixedFft(AVX2)` outperforms `MicroFftBackend` across `N = 128..=8192` (up to `1.28×` at `N = 8192`) with `0 B` heap and a single `16,388 B` quarter-wave ROM. `PhastFftBackend` (`std` desktop) reaches `3.18×–3.25×` at `N = 4096..16384`.
- Isolated CMAC calibration (`fft_backends_bench`): `CMAC AVX2` runs at `0.296–0.393 ns/bin` for `bins <= 1025` (L1 cache resident) and steps up to `0.479–0.488 ns/bin` for `bins = 2049..8193` (L1 cache boundary, matching Wefers Table 7.2); `DOT AVX2 (64 taps) = 0.234 ns/tap`, `DOT SSE2 = 0.240 ns/tap`, `DOT Scalar = 0.907 ns/tap`.

### 7.2 16,384-Tap and 65,536-Tap NUPC vs Hybrid (32-Sample Host Blocks, `SimdFixedFftBackend`)

#### SSE2 Baseline (`--features nonuniform`, `n = 4096`, `warmup = 512`, `k = 2`)

| IR Taps | Engine | `mean ± U` (µs) | `s` (µs) | `u_A` (µs) | `p50` (µs) | `p95` (µs) | `p99` (µs / % budget) | `max` (µs) | Workspace |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | `Hybrid(H128/T128) BURST` | `6.19 ± 0.35` | 11.18 | 0.17 | 1.59 | 17.39 | 29.28 (4.39%) | 395.88 | 261.49 KiB |
| 16,384 | `Hybrid(H128/T128) SPREAD` | `6.29 ± 0.42` | 13.58 | 0.21 | 5.50 | 6.21 | 29.46 (4.42%) | 602.40 | 261.49 KiB |
| 16,384 | `Hybrid(H256/T256) SPREAD` | **`5.44 ± 0.20`** | 6.31 | 0.10 | 4.32 | 9.02 | **14.78 (2.22%)** | 216.32 | 263.99 KiB |
| 16,384 | `Hybrid(H512/T512) SPREAD` | `6.40 ± 0.16` | 5.24 | 0.08 | 5.20 | 17.42 | 22.62 (3.39%) | 102.19 | 270.49 KiB |
| 16,384 | `NUPC default (D32/32/64/128)` | `10.38 ± 0.22` | 7.05 | 0.11 | 9.71 | 15.30 | 40.39 (6.06%) | 137.71 | 269.45 KiB |
| 16,384 | `NUPC large (D32/32/128/512/2k/4k)` | `10.05 ± 0.90` | 28.70 | 0.45 | 4.49 | 26.38 | 135.69 (20.35%) | 477.64 | 495.76 KiB |
| 65,536 | `Hybrid(H128/T128) BURST` | `19.0 ± 2.3` | 72.63 | 1.13 | 1.61 | 60.60 | 100.82 (15.12%) | 2244.31 | 1,035.49 KiB |
| 65,536 | `Hybrid(H128/T128) SPREAD` | `16.64 ± 0.34` | 11.01 | 0.17 | 18.07 | 23.61 | 63.44 (9.52%) | 201.82 | 1,035.49 KiB |
| 65,536 | `Hybrid(H256/T256) SPREAD` | `10.90 ± 0.32` | 10.10 | 0.16 | 9.91 | 11.12 | 42.26 (6.34%) | 462.70 | 1,034.99 KiB |
| 65,536 | `Hybrid(H512/T512) SPREAD` | **`9.15 ± 0.26`** | 8.38 | 0.13 | 7.67 | 17.62 | **39.96 (5.99%)** | 278.76 | 1,039.99 KiB |
| 65,536 | `NUPC default (D32/32/64/128)` | `24.12 ± 0.62` | 19.74 | 0.31 | 20.89 | 40.41 | 95.12 (14.27%) | 509.01 | 1,043.45 KiB |
| 65,536 | `NUPC large (D32/32/128/512/2k/4k)` | `10.49 ± 0.74` | 23.56 | 0.37 | 4.56 | 26.48 | 137.79 (20.67%) | 314.08 | 1,263.95 KiB |
| 65,536 | `NUPC large (D32/32/128/512/2k/8k)` | `10.6 ± 1.4` | 43.69 | 0.68 | **4.25** | 26.16 | 105.08 (15.76%) | 732.70 | 1,407.84 KiB |

#### AVX2+FMA (`--features "nonuniform std"`, `n = 4096`, `warmup = 512`, `k = 2`)

| IR Taps | Engine | `mean ± U` (µs) | `s` (µs) | `u_A` (µs) | `p50` (µs) | `p95` (µs) | `p99` (µs / % budget) | `max` (µs) | Workspace |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,384 | `Hybrid(H128/T128) BURST` | `5.64 ± 0.36` | 11.40 | 0.18 | 1.66 | 12.62 | 62.99 (9.45%) | 339.12 | 261.49 KiB |
| 16,384 | `Hybrid(H128/T128) SPREAD` | `5.6 ± 1.1` | 35.87 | 0.56 | 4.21 | 4.87 | 36.02 (5.40%) | 2273.93 | 261.49 KiB |
| 16,384 | `Hybrid(H256/T256) SPREAD` | **`4.20 ± 0.18`** | 5.65 | 0.09 | 3.13 | 8.28 | **9.10 (1.36%)** | 142.25 | 263.99 KiB |
| 16,384 | `Hybrid(H512/T512) SPREAD` | `4.88 ± 0.14` | 4.62 | 0.07 | 3.81 | 15.63 | 16.30 (2.44%) | 78.69 | 270.49 KiB |
| 16,384 | `NUPC default (D32/32/64/128)` | `9.38 ± 0.18` | 5.64 | 0.09 | 8.81 | 13.63 | 35.34 (5.30%) | 87.53 | 269.45 KiB |
| 16,384 | `NUPC large (D32/32/128/512/2k/4k)` | `9.12 ± 0.72` | 22.92 | 0.36 | 4.61 | 23.65 | 92.29 (13.84%) | 296.42 | 495.76 KiB |
| 65,536 | `Hybrid(H128/T128) BURST` | `11.40 ± 0.56` | 17.99 | 0.28 | 1.66 | 37.92 | 74.33 (11.15%) | 178.58 | 1,035.49 KiB |
| 65,536 | `Hybrid(H128/T128) SPREAD` | `12.13 ± 0.25` | 8.10 | 0.13 | 12.98 | 14.29 | 50.48 (7.57%) | 166.80 | 1,035.49 KiB |
| 65,536 | `Hybrid(H256/T256) SPREAD` | `7.77 ± 0.16` | 5.27 | 0.08 | 6.92 | 8.76 | 41.06 (6.16%) | 126.53 | 1,034.99 KiB |
| 65,536 | `Hybrid(H512/T512) SPREAD` | **`6.72 ± 0.18`** | 5.64 | 0.09 | 5.50 | 15.88 | **37.86 (5.68%)** | 96.08 | 1,039.99 KiB |
| 65,536 | `NUPC default (D32/32/64/128)` | `17.82 ± 0.32` | 10.12 | 0.16 | 16.28 | 25.68 | 63.87 (9.58%) | 148.29 | 1,043.45 KiB |
| 65,536 | `NUPC large (D32/32/128/512/2k/4k)` | `9.72 ± 0.58` | 18.56 | 0.29 | 4.61 | 23.75 | 97.57 (14.64%) | 184.02 | 1,263.95 KiB |
| 65,536 | `NUPC large (D32/32/128/512/2k/8k)` | `10.8 ± 1.3` | 42.86 | 0.67 | **4.46** | 23.55 | 91.86 (13.78%) | 1589.08 | 1,407.84 KiB |

**Honest Interpretation**:
- Unlocking `N512..=N16384` immediately cuts `Hybrid SPREAD` mean latency at 65,536 taps by `1.8×` (`16.64 → 9.15 µs` SSE2; `12.13 → 6.72 µs` AVX2+FMA) and cuts NUPC mean latency at 65,536 taps by `1.8×–2.3×` (`24.12 → 10.49 µs` SSE2; `17.82 → 9.72 µs` AVX2+FMA) with `p50` dropping to **`4.25–4.46 µs`** (beats every Hybrid configuration at median).
- Two bottlenecks remain visible in the PR-A data and directly motivate the next two stacked PRs:
  1. The hand-picked 6-stage Gardner schedule opens 5 FDL stages whose FFT overhead exceeds the optimal 3–4-stage partition (solved by Issue #15 / PR-B's cost-model Viterbi optimizer).
  2. Monolithic `N8192` / `N16384` transforms fire inside a single callback, producing `p99 = 91.9–137.8 µs` and requiring `85.25 KiB` of `D_s + B_s` delay rings (solved by Issue #16 / PR-C's time-distributed FFT/IFFT and clearance-based ring reduction).

### 7.3 AES17 Digital Audio Verification (`997 Hz` @ `48 kHz`, `tests/aes17_audio_metrics.rs`, `HOST_TESTED`)

- **Reference ceiling (16-bit Q1.15 full-scale sine)**: $\mathrm{SNR}_q = 6.02 \cdot 15 + 10.8 - 20 \log_{10}(\sqrt{2}) = 98.09\text{ dB}$ (`97.34 dB` at `−1 dBFS`).
- **Headroom (ITU-R BS.1770-5 Annex 2)**: All filtered test IRs are explicitly $\ell_1$-normalized ($\|h\|_1 \le 0.85$) and verified for zero saturation, zero wrap-around, and `peak_out <= 1.0`.

| Format / Engine | `N` (`log2 N`) | Level | Measured SNR / THD+N | AES17 DR | Ideal Q1.15 Ceiling (Loss) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| `f32` `SimdFixedFft(AVX2)` | `512`–`16384` | `−60 dBFS` | `129.92`–`135.30 dB` | `195.38`–`201.80 dB` | — (24-bit significand) |
| `f32` `SimdFixedFft(AVX2)` | `512`–`16384` | `−1 / 0 dBFS` | `129.41`–`135.25 dB` | `136.11`–`142.29 dB` | — (zero overshoot) |
| `f32` `MicroFftBackend` | `512`–`16384` | `−60 / −1 / 0 dBFS` | `129.07`–`135.61 dB` | `194.56`–`200.95 dB` (`−60`) | — |
| `f32` `PhastFftBackend` | `512`–`16384` | `−60 / −1 / 0 dBFS` | `127.93`–`134.80 dB` | `195.80`–`200.28 dB` (`−60`) | — |
| `Q1.15` `Partitioned` (`unity_imp`) | `64` (`6`) | `−1 dBFS` | `76.72 dB` | `77.72 dB` | `97.34 dB` (`−20.62 dB`) |
| `Q1.15` `Partitioned` (`unity_imp`) | `128` (`7`) | `−1 dBFS` | `73.79 dB` | `74.79 dB` | `97.35 dB` (`−23.55 dB`) |
| `Q1.15` `Partitioned` (`unity_imp`) | `256` (`8`) | `−1 dBFS` | `70.74 dB` | `71.74 dB` | `97.34 dB` (`−26.61 dB`) |
| `Q1.15` `Partitioned` (`unity_imp`) | `512` (`9`) | `−1 dBFS` | `67.45 dB` | `68.45 dB` | `97.35 dB` (`−29.90 dB`) |
| `Q1.15` `Partitioned` (`unity_imp`) | `1024` (`10`) | `−1 dBFS` | `64.96 dB` | `65.97 dB` | `97.34 dB` (`−32.38 dB`) |
| `Q1.15` `Partitioned` (`unity_imp`) | `2048` (`11`) | `−1 dBFS` | `63.21 dB` | `64.21 dB` | `97.34 dB` (`−34.14 dB`) |
| `Q1.15` `Partitioned` (`l1_fir`) | `64`–`2048` | `−60 dBFS` | `34.67`–`36.34 dB` | `97.87`–`98.65 dB` | `97.87`–`98.65 dB` (`0.00 dB`) |
| `Q1.15` `Partitioned` (`l1_fir`) | `2048` (`11`) | `−1 dBFS` | `69.31 dB` | `73.72 dB` | `93.70 dB` (`−24.38 dB`) |
| `Q1.15` `NonUniform` (up to `N2048`) | `64..=2048` | `−60 dBFS` | `34.90 dB` | `98.35 dB` | `98.35 dB` (`0.00 dB`) |
| `Q1.15` `NonUniform` (up to `N2048`) | `64..=2048` | `−1 dBFS` | `81.56 dB` | `86.02 dB` | `93.55 dB` (`−11.99 dB`) |

**Honest Numerical Note (`Q1.15` BFP scaling loss)**: Across `N = 64 → 2048` (`log2(N) = 6 → 11`), `Q1.15` `unity_imp` loses `13.51 dB` over 5 additional radix-2 stages (`~2.7 dB/stage`, matching the expected `~0.5 bit/stage` BFP butterfly truncation/rounding budget across forward + inverse transforms in 16-bit mantissas). At `−60 dBFS`, input-block BFP normalization preserves the full `~98 dB` 16-bit output dynamic range (`0.00 dB` loss vs the Q1.15 output ceiling). Reaching `> 90 dB` THD+N at `-1 dBFS` for `N = 2048` in fixed point requires 32-bit BFP mantissas (tracked in Issue #23).
