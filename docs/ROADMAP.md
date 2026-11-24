# Roadmap (`docs/ROADMAP.md`)

Evolution plan for `rtconv-simd` after `0.2.0`. Status labels keep three things apart: what is implemented on `main`, what is open, and what still needs physical hardware verification. Tracking issue: [#13](https://github.com/SM-Acoustics/rtconv-simd/issues/13).

**Primary reference**: F. Wefers, *Partitioned convolution algorithms for real-time auralization*, PhD thesis, RWTH Aachen University, Logos Verlag Berlin, 2015 (ISBN 978-3-8325-3943-6). § numbers below refer to this thesis.

---

## 1. Current state (`0.2.0`)

| Area | Status |
| :--- | :--- |
| `f32` engines: Direct, uniform OLS FDL, zero-latency Hybrid (BURST / SPREAD), multi-stage NUPC (≤ 8 stages) | Implemented, `HOST_TESTED` (f64-oracle SNR > 100 dB) |
| Q1.15 / block-floating-point engines (Direct, FDL, NUPC) with per-partition coefficient exponents | Implemented, within ~1.7 dB of the Q1.15 output ceiling |
| FIR artifact decoder (WAV / raw / text) and zero-config `AutoConvolver` (compile-time FPU domain: ESP32-P4 → `f32`, ESP32-C6 → Q1.15) | Implemented; selects cost-optimized NUPC (`NonUniform` / `FixedNonUniform`) automatically when cheaper than `Hybrid` (R2 / #15) |
| IEEE 754 subnormal guard (`simd_kernels::DENORMAL_GUARD = 2^-63`) | Implemented. Measured on x86 without FTZ/DAZ: up to 80× slowdown removed |
| FFT sizes | `N64..=N16384` (`B = 32..=8192`) in all `f32` backends; `N64..=N2048` (`B = 32..=1024`) in `fft_q15` (R1 / #14, `HOST_TESTED`, `CROSS_COMPILE_VERIFIED`, `DEVICE_VERIFIED = NO`) |
| NUPC partition optimizer | Zero-alloc `#![no_std]` Viterbi DP (`CostModel`, `ClearancePolicy`, `NonUniformPlan::optimize`, block-mode Wefers validation, MCU presets; R2 / #15, `HOST_TESTED`, `CROSS_COMPILE_VERIFIED`, `DEVICE_VERIFIED = NO`) |
| Device execution (ESP32-C6 / ESP32-P4) | `CROSS_COMPILE_VERIFIED`, `DEVICE_VERIFIED = NO` |

---

## 2. Research-backed roadmap (Wefers 2015)

| # | Item | Thesis basis | Issue | Status | Depends on |
| :-: | :--- | :--- | :---: | :---: | :---: |
| R1 | **FFT sizes `N512`–`N16384`** (f32), up to `N2048` in `fft_q15` | §8.2: non-uniform partitions pay off for `N > 20·B`. §6.12: 88,200 taps at B=128 cost 240 vs 2,018 cycles/sample (8.4×) | [#14](https://github.com/SM-Acoustics/rtconv-simd/issues/14) | `VERIFIED (HOST_ONLY)` / `CROSS_COMPILE_VERIFIED` | — |
| R2 | **Partition optimizer**: García DAG + Viterbi with minimum-clearance restrictions and a benchmark-calibrated `CostModel` | §6.5.2 states/weights (Eq. 6.41–6.45); §6.5.3 minimal-load partitions are impractical (zero clearance); §6.5.4 strategies 1–2; Ch. 7 cost models | [#15](https://github.com/SM-Acoustics/rtconv-simd/issues/15) | `VERIFIED (HOST_ONLY)` / `CROSS_COMPILE_VERIFIED` | R1 |
| R3 | **Time-distributed FFT/IFFT** and even per-callback load on single-core MCUs | §6.8 manual preemption and subdivision of transforms (Hurchalla); §6.4.6 Gardner time spans | [#16](https://github.com/SM-Acoustics/rtconv-simd/issues/16) | Open | R1 |
| R4 | **Glitch-free filter exchange**: FD crossfade (UPOLS) and coherent / asynchronous NUPC exchange | §4.4.1 TD crossfade (+50–60 %); §4.4.2 FD crossfade, 3-coefficient operator (+17–34 %); §6.11 Müller-Tomfelde strategies | [#17](https://github.com/SM-Acoustics/rtconv-simd/issues/17) | Open | — |
| R5 | **Frequency-domain MIMO**: shared input spectra, crossover networks, true stereo, dual-channel complex FFT | §4.2 / §5.5 transform once per input and output; §4.3 parallel assemblies (2.0–4.1×); §4.2.1 complex-FFT dual channel (1.56× at ≤ 128 taps); §6.9 NUPC MIMO | [#18](https://github.com/SM-Acoustics/rtconv-simd/issues/18) | Open | — |
| R6 | **Opt-in perceptual spectral pruning** (lossy, research) | §6.13 Lee et al. / Primavera et al. (~30 % speed-up) | [#22](https://github.com/SM-Acoustics/rtconv-simd/issues/22) | Open | S2 (optional) |

### Considered and not planned (thesis evidence)

- **Exploiting symmetric / linear-phase filters** (§2.5.3): the gains are marginal, and partitioning destroys the symmetry. rePhase linear-phase FIRs therefore get no special path.
- **Partial DFTs / pruned FFTs** (§4.5, §8.2): too little saving for the implementation burden.
- **Non-power-of-two transform sizes for UPOLS** (§8.2): the standard `L = B`, `K = 2B` is optimal or near-optimal for power-of-two `B`.
- **Karatsuba short convolution for the direct head** (§2.4, §8.2): only theoretically better below 128 taps, and the thesis left it unverified. It can be revisited after on-device profiling (#24).

---

## 3. Heterogeneous compute, standards and platform items

| # | Item | Basis | Issue |
| :-: | :--- | :--- | :---: |
| H1 | **Heterogeneous execution.** The CPU keeps the head and short segments (zero latency). Long-clearance tail segments go through a `no_std` deadline-based handoff to a second core (ESP32-P4, ESP32-S3) or to a GPU executor in a separate `std` crate (`wgpu`). | Pure GPU low-latency audio fails because bus transfer and kernel launch cost more than the CPU work for tiny blocks. Basis: clearance model §6.4.3; §6.8 multi-processor scheduling; Battenberg & Avizienis (EDF, segment-per-core). | [#19](https://github.com/SM-Acoustics/rtconv-simd/issues/19) |
| S1 | **IEEE 754 (FP32/FP64) subnormals** | Implemented in 0.2.0 (see §4). Remaining: on-device confirmation for the ESP32-P4 FPU, where RISC-V F subnormal speed is implementation-defined. | [#24](https://github.com/SM-Acoustics/rtconv-simd/issues/24) |
| S2 | **ISO 3382-1 IR analysis** (Schroeder EDC, T20/T30/EDT, C50/C80, D50), `no_std` | ISO 3382-1:2009 / ISO 3382-2 parameter definitions. This is an analysis utility, not a certified measurement. | [#20](https://github.com/SM-Acoustics/rtconv-simd/issues/20) |
| S3 | **AES69 (SOFA) import** via a companion `std` tool, plus a zero-copy `FirBank` in the core | AES69-2022 (SOFA 2.1) is netCDF-4 / HDF5 and cannot reasonably be parsed in a `no_std` core | [#21](https://github.com/SM-Acoustics/rtconv-simd/issues/21) |
| P1 | **Q1.15 follow-ups**: 32-bit-mantissa BFP, real-input packing, caller-owned accumulators | `docs/REVIEW-2026-11-ADVERSARIAL.md` residual risks | [#23](https://github.com/SM-Acoustics/rtconv-simd/issues/23) |
| P2 | **Device validation and calibrated `auto` thresholds** (ESP32-C6 / ESP32-P4, `mcycle`) | Ch. 7: benchmark-based cost models (theory/practice ratios differ) | [#24](https://github.com/SM-Acoustics/rtconv-simd/issues/24) |
| P3 | **Immutable shareable IR spectra** (`&'static` ROM/Flash) for `f32` and Q1.15 | Partition spectra are already isolated (`ir_re` / `ir_im` / `ir_exp`) | — |

---

## 4. IEEE 754 subnormal policy (implemented)

Subnormal ("denormal") operands and results are executed by microcode assists on x86 unless the host thread enables FTZ/DAZ. Rust never enables them, and `#![forbid(unsafe_code)]` rules out writing `MXCSR`. Measured with `cargo bench --bench denormal_bench` (x86-64 SSE2 container, 1024 taps, 32-sample blocks):

| Case | `DirectConvolver` before → after | `HybridConvolver` before → after |
| :--- | ---: | ---: |
| Normal input (0.1) | 4.57 → 4.29 µs | 0.93 → 1.01 µs |
| Input 1e-35 (normal, products subnormal) | 142.8 → 4.11 µs | 1.57 → 1.02 µs |
| Input 1e-37 | 346.3 → 4.33 µs | 48.1 → 0.92 µs |
| Input 1e-39 (subnormal) | 342.0 → 4.15 µs | 47.7 → 0.96 µs |
| Deep-tail IR (≈150 subnormal `f32` taps from an f64 design), input 0.1 | 67.4 → 4.15 µs | 1.07 → 0.92 µs |

The policy keeps non-zero input samples, coefficients and transformed spectra at `|v| ≥ 2^-63`. Every non-zero product is then `≥ 2^-126` (normal). FDL slots, accumulators and outputs are flushed as well, because cancellation can still leave subnormal residues there. The flushed magnitudes are about −379 dBFS. Hosts can still enable FTZ/DAZ for complete coverage. Q1.15 engines are integer-only and unaffected.

---

## 5. Release milestones

### `0.1.0`: GitHub release `v0.1.0`
- `#![no_std]`, `#![forbid(unsafe_code)]`, caller-owned `&mut [f32]` workspaces.
- `DirectConvolver`, `PartitionedConvolver` (`N64/N128/N256`), zero-latency `HybridConvolver` with `SPREAD`.
- `MicroFftBackend`, `SimdFixedFftBackend`, optional `PhastFftBackend`.

### `0.2.0`: version on `main`
- Work orders `WO-RTCONV-004A…F`: CI hardening, `f32` NUPC, Q1.15/BFP stack, ESP32-C6 harness.
- Adversarial review (`docs/REVIEW-2026-11-ADVERSARIAL.md`): true BFP, single-rounding fixed NUPC, honest budget model, per-partition Q1.31 coefficients.
- `artifact` decoder, `auto` layer, IEEE 754 subnormal guard.
- The `v0.2.0` tag, GitHub release and crates.io publication are **maintainer actions**; see `CHANGELOG.md`.

### `0.3.0`: planned
- R1 (#14), R2 (#15), R3 (#16) as the performance core for long IRs on MCUs.
- R4 (#17) and R5 (#18) for the speaker-DSP / binaural feature set.
- P1 (#23) and P2 (#24) for Q1.15 precision and device evidence.
