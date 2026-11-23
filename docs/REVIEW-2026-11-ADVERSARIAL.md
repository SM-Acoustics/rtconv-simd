# Adversarial DSP Review — `rtconv-simd` (2026-11-23)

**Scope**: every engine, FFT backend, fixed-point path, scheduler, budget model, document claim and CI
gate on `main` at `fa0c0d8`, checked against the partitioned-convolution and fixed-point FFT
literature and against independent `f64` oracles written for this review.
**Provenance**: `HOST_TESTED` (Linux x86-64 container, `rustc 1.97.0` stable and `1.89.0` MSRV),
`CROSS_COMPILE_VERIFIED` (`riscv32imac-unknown-none-elf`, `riscv32imafc-unknown-none-elf`,
`thumbv6m-none-eabi`, `thumbv7em-none-eabihf`, plus the `hardware/esp32c6` harness).
`DEVICE_VERIFIED = NO`: nothing was flashed or run on ESP32-C6/P4 hardware.
**Method**: adversarial tests first (`tests/adversarial_reference.rs`), fix only what the tests or
the literature prove wrong, then re-measure. AI-assisted review; independent human review requested.

---

## 1. Findings

Severity: **Critical** = wrong results or a false guarantee in a core claim; **High** = a documented
contract that does not hold; **Medium** = misleading documentation or missing verification;
**Low** = polish.

| # | Severity | Area | Finding | Evidence | Resolution |
| :-: | :--- | :--- | :--- | :--- | :--- |
| F1 | **Critical** | `fixed::fft_q15` | The "block floating point" FFT only **down-scaled**; quiet blocks were never normalized up, so precision fell with signal level, the behaviour BFP exists to avoid. SNR was 47.8 dB at 0 dBFS, 46.0 dB at −20 dBFS and 30.5 dB at −40 dBFS (partitioned), 47.7 / 43.0 / 25.8 dB (NUPC). | `fixed_engines_snr_vs_input_level`; Kalliojärvi & Astola (1996): BFP SNR should stay roughly constant over a wide dynamic range | Input-block normalization in `rfft_q15_bfp` / `irfft_q15_bfp` (negative block exponents), with per-stage conditional down-scaling kept. **Partitioned: 68.7 / 50.4 / 30.5 dB**, within 1.7 dB of the Q1.15 output-rounding ceiling (70.4 / 50.4 / 30.5 dB). |
| F2 | **Critical** | `FixedQ15NonUniformConvolver` | Every stage's output was rounded to Q1.15 before summing, which adds `10·log10(stages)` ≈ 6 dB of noise, and the direct head was saturated on its own before the sum. | Same test: NUPC sat 6 dB under partitioned. | FDL stage outputs are kept at Q1.23 (`i32`, two `i16` words) in the delay rings, and the direct head is summed at Q2.30. Rounding to Q1.15 now happens once. **NUPC: 69.4 / 50.4 / 30.5 dB**, within 1 dB of the ceiling. |
| F3 | **High** | Fixed-point coefficients | Long FIRs were quantized on the absolute Q1.15 grid (LSB = −90 dBFS), so the low-level tails of rePhase-style linear-phase filters kept only a few bits. | `q31_partition_coefficients_beat_absolute_q15_taps`: 4096-tap linear-phase FIR measured **62.6 dB** on the original code. | New `prepare_q31` / `prepare_q31_with` store each FFT partition with **its own block exponent**, so the source keeps 15 significant bits relative to the partition peak. Same scenario: **70.4 dB** (64.2 dB with Q1.15 taps after F1). |
| F4 | **High** | `NonUniformPlan` budget | `estimated_worst_ops_per_callback` was documented as a "conservative upper bound" but it counted **one** FDL block per callback whatever the host block size, and only one transform per block. A 512-sample callback with `B = 32` runs 16 blocks. | `budget_estimate_scales_with_host_block` failed: 4,508 vs 19,868 ops for 32 vs 512 frames. | Per-stage bound `fdl_stage_worst_ops`: `ceil(H/B)` triggers, FFT plus IFFT at `5·N·log2 N` each, an in-flight spread completion, and spread steps bounded by step interval. The model is documented as an operation count, not WCET. |
| F5 | **High** | `FixedQ15NonUniformConvolver` memory | About 16.5 KiB of `i64` spectral accumulators (`2 × 8 × 129 × 8 B`) lived **inside the struct**. They were missing from `memory_requirements` and from `NonUniformBudget::max_memory_bytes` admission, and the "~50 % of f32 RAM" claim ignored them. This is a stack-overflow risk on MCU tasks. | Source inspection; `size_of` | Added `FixedQ15NonUniformMemoryRequirements::instance_bytes`. It is included in budget admission and documented in `MEMORY_MODEL.md`. |
| F6 | **Medium** | `NonUniformConvolver` (`f32`) | The direct head ignored the selected backend (`let _ = backend;`) and ran a reversed scalar loop, but `ARCHITECTURE.md` claimed a "SIMD FIR head". | Source inspection | The head now uses the same reverse-indexed circular buffer and `dot_two_segments_simd` as `DirectConvolver`. Gain is marginal (mean 2.75 → 2.63 µs at 2,048 taps, within run-to-run noise) because per-sample scheduling dominates; this is now documented honestly. |
| F7 | **Medium** | Q1.15 FFT twiddles | `W⁰ = 1` and `W^{N/4} = −j` were applied as `32767/32768`, a systematic −2.6·10⁻⁴ dB gain error per affected butterfly. | Table inspection | The `i32` butterflies apply them as exact `±32768`. The overflow bound is proven in the source comment: `|W·B| ≤ √2·2³⁰ < 2³¹`. |
| F8 | **Medium** | Auto FPU detection | On stable Rust, `cfg(target_feature = "f")` is **not** exposed for RISC-V. `riscv32imafc` (ESP32-P4) and `riscv32imac` (ESP32-C6) report identical cfgs, so any cfg-only FPU detection is wrong for the two target chips. | `rustc --print cfg --target …` | `build.rs` derives `cfg(rtconv_no_fpu)` from the triple's ISA string (`f`/`d`/`g`), the ARM `eabihf` ABI and VFP features, and AVR/MSP430/ESP32-S2. Probed per target (§3). |
| F9 | **Medium** | CI coverage | `embedded.yml` ran `cargo check --lib --no-default-features` only. The fixed-point features and the ESP32-C6 harness, both reported as `CROSS_COMPILE_VERIFIED`, were never compiled by CI, and ESP32-P4 (`riscv32imafc`) had no target at all. | `.github/workflows/embedded.yml` | Matrix adds `riscv32imafc-unknown-none-elf` and `thumbv6m-none-eabi`, builds default (`auto`) and `fixed-nonuniform` with `-D warnings`, and builds `hardware/esp32c6`. |
| F10 | **Medium** | Docs vs code | (a) README listed `PartitionedConvolver` latency as `B` while `capabilities()` reports `0` (framing latency only when host blocks are smaller than `B`). (b) The FFT table gave `fft_q15` peak stack as "minimal", but it is two `[i32; 256]` (2 KiB) per call plus a 2 KiB `i64` accumulator in `FixedQ15PartitionedConvolver`. (c) `COMPETITIVE_ANALYSIS.md` still listed non-uniform partitioning as roadmap. (d) The ESP32-C6 harness header claimed it exercised the NUPC, but it did not. | Cross-reading | All corrected. The harness now runs `AutoConvolver`, which selects the Q1.15 NUPC, from an embedded raw `.bin` artifact, mirrored by a host test. |
| F11 | **Gap** | Product goal | No way to load the FIR artifacts that design tools export (WAV PCM 16/24/32, IEEE 32/64, mono/stereo, raw `.bin`, text), and no automatic engine or numeric-domain selection. Users had to choose engines, block sizes, workspace layouts and Q15 scaling by hand. | Requirements | New `artifact` module (zero-copy, `no_std`, allocation-free decoder) and `auto` module (`AutoConvolver` / `AutoStereoConvolver` with a cost-model engine choice, compile-time FPU domain, gain headroom, and `const fn` workspace sizing). |
| F12 | **Low** | Tests | Zero unit tests, zero doctests, and no sweep over IR lengths that straddle partition boundaries with fragmented host blocks for every backend. | `cargo test` output | `tests/adversarial_reference.rs` sweeps 17 IR lengths × scalar/SIMD × MicroFFT/SimdFixedFFT × B = 32/64/128 × BURST/SPREAD × random host blocks of 1–3B against an `f64` oracle (SNR > 100 dB required), plus NUPC custom plans with slack. A doctest was added in `auto`. |

### Follow-up finding (same day, after merge of the review)

| # | Severity | Area | Finding | Evidence | Resolution |
| :-: | :--- | :--- | :--- | :--- | :--- |
| F13 | **High** | IEEE 754 subnormals, all `f32` engines | On x86-64 without FTZ/DAZ (Rust's default), subnormal arithmetic is microcoded, so the cost depended on the signal: (a) decaying or near-silent inputs, (b) tiny normal inputs whose products with the taps are subnormal, (c) an f64 FIR design with a deep tail, which decodes to subnormal `f32` taps. Case (c) is a permanent slowdown. | `benches/denormal_bench.rs` (1024 taps, 32-sample blocks): Direct 4.57 µs normal vs 142.8 / 346.3 / 342.0 µs for 1e-35 / 1e-37 / 1e-39 inputs, and 67.4 µs for the deep-tail IR; Hybrid 0.93 vs 48.1 µs. | `simd_kernels::DENORMAL_GUARD = 2^-63` applied to inputs, coefficients, IR spectra, FDL slots, accumulators and outputs. Guarded × guarded products are `≥ 2^-126`, which is normal. **All cases now run at 4.1–4.3 µs (Direct) and 0.9–1.0 µs (Hybrid).** `tests/denormal_guard.rs`. |

### Verified correct (no change needed)

- `f32` engines: `DirectConvolver`, `PartitionedConvolver` (50 % overlap-save FDL, Soo & Pang 1990 / Kulp 1988),
  `HybridConvolver` (Gardner 1995 zero-delay head + tail, BURST and SPREAD time distribution in the
  sense of Battenberg & Avizienis 2011), and `NonUniformConvolver` match the `f64` oracle at
  SNR > 100 dB for every swept length, backend and host fragmentation.
- `SimdFixedFftBackend`: an `N/2`-point complex FFT with real-data untangling (Sorensen et al. 1987); forward and inverse
  match the oracle.
- The NUPC causality condition `D_s ≥ B_s`, the `min(slack, B − 1)` spread window and the delay-ring capacity
  `D_s + B_s` were checked analytically and empirically (no deadline miss, no ring overrun).
- `#![no_std]`, `#![forbid(unsafe_code)]`, and zero heap activity in prepare, process and reset (standalone
  `GlobalAlloc` harness, now also covering artifact parsing and `AutoConvolver`).

---

## 2. Before / after measurements (`HOST_TESTED`, Linux x86-64 container)

| Metric | Before (`fa0c0d8`) | After | Ceiling / note |
| :--- | ---: | ---: | :--- |
| `FixedQ15Partitioned` SNR, 0 / −20 / −40 dBFS input | 47.8 / 46.0 / 30.5 dB | **68.7 / 50.4 / 30.5 dB** | 70.4 / 50.4 / 30.5 dB (one Q1.15 output rounding) |
| `FixedQ15NonUniform` SNR, 0 / −20 / −40 dBFS input | 47.7 / 43.0 / 25.8 dB | **69.4 / 50.4 / 30.5 dB** | same ceiling |
| 4096-tap linear-phase FIR (rePhase-style), Q1.15 NUPC | 62.6 dB | 64.2 dB (Q1.15 taps), **70.4 dB** (`prepare_q31`) | 85.2 dB output ceiling; the residual gap is the 16-bit-mantissa FFT path |
| NUPC budget, `host_block_max` 32 → 512 | 4,508 → 19,868 ops (not an upper bound) | scales with `ceil(H/B)` | operation count, not WCET |
| `NonUniform(D32/32/64/128)` `f32` mean, 2,048 / 16,383 taps | 2.75 / 4.41 µs | 2.63 / 4.32 µs | container timing, noisy; marginal |

Commands: `cargo test --release --all-features --test adversarial_reference -- --nocapture`,
`cargo bench --bench nonuniform_bench --features nonuniform` (before = a worktree at `fa0c0d8`).

---

## 3. Automatic mode and target classification

`build.rs` classification, probed with `cargo build -vv --target …`:

| Target | Chip examples | `auto::NativeElem` | Domain |
| :--- | :--- | :---: | :--- |
| `riscv32imac-unknown-none-elf` | ESP32-C6, ESP32-H2 | `i16` | Q1.15 / BFP |
| `riscv32imafc-unknown-none-elf` | ESP32-P4 | `f32` | `f32` (scalar; `fearless_simd` has no RVV/PIE path) |
| `thumbv6m-none-eabi` | Cortex-M0/M0+ | `i16` | Q1.15 / BFP |
| `thumbv7em-none-eabihf` | Cortex-M4F/M7 | `f32` | `f32` |
| `x86_64-unknown-linux-gnu` | desktop | `f32` | `f32` with SSE2/AVX2 |

Engine selection: in `f32`, the cheapest of `Direct` (`L` MACs/sample) and the hybrid
`B + 4L/B + 20·log2(2B)` for `B ∈ {32, 64, 128}`, which is Garcia's (2002) cost-model
approach restricted to the FFT sizes the crate supports. In Q1.15, direct up to 128 taps,
otherwise the default 4-stage NUPC. Q1.15 thresholds are **operation-count heuristics** that have not
yet been measured on device.

---

## 4. Residual risks and open items

1. **No device execution.** The ESP32-C6 harness compiles but has never been flashed. Cycle counts,
   stack high-water marks and real-time margins on ESP32-C6/P4 are unknown (`OPEN`).
2. **The Q1.15 FFT path tops out near 70 dB SNR** because the butterflies use 16-bit mantissas. A 32-bit-mantissa
   BFP (i32 × Q15 → i64 twiddle products) would close the gap to the output ceiling. This is on the roadmap.
3. **`fft_q15` runs a full `N`-point complex FFT on real data.** The `N/2`-point packing trick already used by
   `SimdFixedFftBackend` would halve its cost on ESP32-C6. This is on the roadmap.
4. **FFT sizes are capped at `N = 256`,** so very long IRs (≥ 16k taps) are compute-heavy on MCUs (see `ROADMAP.md`).
5. **Raw `.bin` auto-detection is heuristic.** The f32/f64 plausibility scoring rejects random data and handles
   exact-valued f64, but raw integer files carry no type information and require `parse_raw`.
6. **rePhase's export dialog is not formally documented.** Decoding is implemented from the RIFF/WAVE and
   `WAVEFORMATEXTENSIBLE` specifications and synthetic fixtures for the formats listed in the
   requirements. No real rePhase-exported file was available in this environment.
7. **The fixed NUPC struct holds about 17 KiB inline,** so keep it in a `static`. Moving the accumulators into the caller
   workspace would need an `i64` workspace slice, which is an API change deferred to 0.3.

---

## 5. References

- W. G. Gardner, "Efficient Convolution without Input-Output Delay", *JAES* 43(3):127–136, 1995.
- G. García, "Optimal Filter Partition for Efficient Convolution with Short Input/Output Delay",
  AES 113th Convention, paper 5660, 2002.
- F. Wefers, *Partitioned Convolution Algorithms for Real-Time Auralization*, PhD thesis, RWTH Aachen, 2015.
- E. Battenberg, R. Avizienis, "Implementing Real-Time Partitioned Convolution Algorithms on
  Conventional Operating Systems", DAFx-11, 2011.
- J.-S. Soo, K. K. Pang, "Multidelay Block Frequency Domain Adaptive Filter", *IEEE TASSP* 38(2), 1990.
- T. G. Stockham, "High-Speed Convolution and Correlation", AFIPS SJCC, 1966.
- H. V. Sorensen, D. L. Jones, M. T. Heideman, C. S. Burrus, "Real-Valued Fast Fourier Transform
  Algorithms", *IEEE TASSP* 35(6), 1987.
- P. D. Welch, "A Fixed-Point Fast Fourier Transform Error Analysis", *IEEE Trans. Audio
  Electroacoust.* 17(2), 1969.
- A. V. Oppenheim, C. J. Weinstein, "Effects of Finite Register Length in Digital Filtering and the
  Fast Fourier Transform", *Proc. IEEE* 60(8), 1972.
- K. Kalliojärvi, J. Astola, "Roundoff Errors in Block-Floating-Point Systems", *IEEE Trans. SP*
  44(4), 1996.
- Texas Instruments SPRA948, "A Block Floating Point Implementation for an N-Point FFT on the
  TMS320C55x DSP", 2003.
- Microsoft, "WAVEFORMATEXTENSIBLE" and "Multiple Channel Audio Data and WAVE Files" (KSDATAFORMAT
  sub-format GUIDs, 24-bit packing, `wValidBitsPerSample`).
- Reference implementations consulted for architecture comparison: HiFi-LoFi `FFTConvolver` /
  `TwoStageFFTConvolver`, `neodsp/fft-convolver`, Zita-convolver, BruteFIR, ARM CMSIS-DSP
  `arm_fir_q15` (64-bit accumulation) and `arm_rfft_q15`.
