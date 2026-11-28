# RTConv SIMD (`rtconv-simd`)

**Zero-latency, zero-allocation FIR convolution in pure Rust, from AVX2/NEON desktops to FPU-less RISC-V microcontrollers. Give it a FIR file and it picks the engine.**

`#![no_std]` · `#![forbid(unsafe_code)]` · no heap (`alloc` is never used) · MSRV `1.89.0` · `GPL-3.0-or-later`

`rtconv-simd` (library crate `rtconv_simd`) is a real-time FIR convolution suite with:

- caller-owned workspaces,
- pluggable SIMD FFT backends,
- multi-stage non-uniform partitioning (NUPC),
- a Q1.15 / block-floating-point (BFP) fixed-point stack for chips without an FPU,
- an `auto` layer that loads FIR artifacts (WAV, raw `.bin` or text, as written by rePhase and similar tools) and configures itself.

---

## Why RTConv SIMD?

- **No configuration**: `AutoConvolver::from_bytes(include_bytes!("filter.wav"), 0, ws)` parses the artifact and chooses everything else:
  - the numeric domain, at compile time: `f32` where the target has a hardware FPU, Q1.15/BFP where it does not;
  - the engine and partition size, from an operation-count cost model;
  - Q1.15 gain headroom, applied automatically.
- **Strict real-time core**: `#![no_std]`, `#![forbid(unsafe_code)]`, no heap at all.
  - Every buffer is a caller slice, sized by checked arithmetic (or by a `const fn` usable in `static`s).
  - A standalone `GlobalAlloc` harness checks for **0 allocations** in artifact parsing, `prepare()`, `process_*()` and `reset()`.
- **Zero algorithmic latency on long IRs**: a direct-form head is combined with partitioned FFT tails (Gardner 1995), either in `HybridConvolver` (with time-distributed `SPREAD`) or in the multi-stage `NonUniformConvolver`.
- **Desktop SIMD and FPU-less MCUs**:
  - `f32` is accelerated with `fearless_simd 1.2.0` (SSE2, SSE4.2, AVX2+FMA, NEON, WASM SIMD128).
  - The Q1.15 stack uses true block-floating-point FFTs. Its SNR stays within about 2 dB of the Q1.15 output-rounding limit across input levels (see [benchmarks](#numerical-accuracy-host_tested)).
- **Phase-agnostic and bit-honest**: any finite causal FIR is processed as given. Tap order, gain, phase and delay are preserved, with no hidden normalization or windowing. The one exception is that magnitudes below `2^-63` (≈ −379 dBFS) are flushed to zero to stay out of IEEE 754 subnormal arithmetic (see [Real-Time Guarantees](#real-time-guarantees-and-limits)).
- **Evidence-labelled claims**: `HOST_TESTED`, `CI_VERIFIED`, `CROSS_COMPILE_VERIFIED` and `DEVICE_VERIFIED` are kept separate. There are no synthetic cycle counts and no WCET promises. See the [adversarial review](./docs/REVIEW-2026-11-ADVERSARIAL.md).

---

## Quick Start: automatic

```rust
use rtconv_simd::auto::{AutoConvolver, AutoWorkspace, workspace_len};

// rePhase export: WAV (PCM 16/24/32, IEEE 32/64, mono/stereo), raw .bin (f32/f64) or text.
let fir: &[u8] = b"0.5\n0.25\n-0.125\n"; // e.g. include_bytes!("speaker_correction.wav")

// One const-sized workspace in the target's native element type (f32 or i16).
let mut ws = AutoWorkspace::<{ workspace_len(4096) }>::new();
let mut conv = AutoConvolver::from_bytes(fir, 0, ws.as_mut_slice()).unwrap();

let input = [1.0f32, 0.0, 0.0, 0.0];
let mut output = [0.0f32; 4];
conv.process_f32(&input, &mut output).unwrap(); // or process_i16 for I2S PCM
assert!((output[1] - 0.25).abs() < 1e-3);
println!("{:?}", conv.info()); // domain, engine, taps, latency = 0, gain shift, sample rate
```

Stereo artifacts use `AutoStereoConvolver` (planar `process_f32` / `process_i16`, or `process_interleaved_i16` for I2S DMA buffers). A mono artifact is applied to both channels. See [`examples/auto_from_artifact.rs`](./examples/auto_from_artifact.rs).

| Target | Example chips | Native domain | Engines selected |
| :--- | :--- | :--- | :--- |
| `riscv32imac-unknown-none-elf` | **ESP32-C6**, ESP32-H2 | Q1.15 / BFP (no FPU) | `FixedQ15Direct` (≤ 128 taps) or cost-optimized `FixedQ15NonUniform` (`B ≤ 1024`) |
| `riscv32imafc-unknown-none-elf` | **ESP32-P4** | `f32` (hardware FPU, scalar) | `Direct`, `Hybrid` (`B` = 32..=8192), or cost-optimized `NonUniform` |
| `thumbv6m-none-eabi`, `thumbv7m-none-eabi` | Cortex-M0/M3 | Q1.15 / BFP | as ESP32-C6 |
| `thumbv7em-none-eabihf` | Cortex-M4F/M7 | `f32` | as ESP32-P4 |
| x86-64 / AArch64 / WASM | desktop, mobile, web | `f32` + SIMD | as ESP32-P4 |

The FPU is detected by `build.rs` from the target ISA string. Stable Rust does not expose the RISC-V `F` extension as a `cfg`. The `native-q15` feature forces the fixed-point domain on any target; `native-q31` forces the Q1.31 domain (`NativeElem = i32`, about twice the workspace bytes of Q1.15, so Cortex-M0-class parts keep the Q1.15 default). FPU-less builds link no `f32` engine, no soft-float routine and no decimal-to-float table in the automatic layer (CI-guarded on the ESP32-C6 harness).

---

## Convolution Engines

All core engines are mono; use one instance per channel, or the `auto` stereo wrapper.

| Engine | Feature | Numeric format | Latency | Best for |
| :--- | :--- | :--- | :---: | :--- |
| **`AutoConvolver` / `AutoStereoConvolver`** | `auto` (default) | `f32` or Q1.15 (by target) | `0` | Loading FIR artifacts with zero configuration |
| **`DirectConvolver`** | core | `f32` (SIMD / scalar) | `0` | Short FIRs (≲ 128 taps by the cost model), crossovers |
| **`PartitionedConvolver`** | core | `f32` (uniform 50 % overlap-save FDL) | `0`¹ | Block-aligned FFT convolution (`B ∈ {32..=8192}`) |
| **`HybridConvolver`** | core | `f32` (direct head + FDL tail, `SPREAD`) | `0` | Medium/long IRs on desktop and FPU MCUs (`B ∈ {32..=8192}`) |
| **`NonUniformConvolver`** | `nonuniform` | `f32` (up to 8 causal stages) | `0` | Long IRs with cost-optimized schedules (`NonUniformPlan::optimize`, `B ∈ {32..=8192}`) |
| **`FixedQ15DirectConvolver`** | `fixed-q15` | Q1.15 (`i64` accumulator) | `0` | FPU-less MCUs, ≤ 1 LSB error, half the RAM of `f32` |
| **`FixedQ15PartitionedConvolver`** | `fixed-fft` | Q1.15 + BFP FFT/FDL | `0`¹ | Medium/long FIRs on FPU-less MCUs (`B ∈ {32..=1024}`) |
| **`FixedQ15NonUniformConvolver`** | `fixed-nonuniform` | Q1.15 + BFP NUPC | `0` | Zero-latency long FIRs on FPU-less MCUs (`B ∈ {32..=1024}`) |

¹ The output for an input block is produced in the same call. A host that delivers fewer than `B` samples per callback adds `B` samples of framing delay, which is why the hybrid and NUPC engines exist.

> **Large FFTs, Viterbi optimizer & time-distributed transforms (#14, #15, #16)**: every `f32` FFT backend supports `N ∈ {64..=16384}` (`B ∈ {32..=8192}`), `fft_q15` supports `N ∈ {64..=2048}` (`B ∈ {32..=1024}`), `NonUniformPlan::optimize` finds globally optimal causal NUPC schedules under `CostModel` and `ClearancePolicy` with zero heap allocation, and `SimdFixedFftBackend` / `fft_q15` time-distribute RFFT/CMAC/IFFT across each stage's causal clearance with power-of-two `2 * B_s` output delay rings. Remaining roadmap tracks (#17–#24) are documented in [`docs/ROADMAP.md`](./docs/ROADMAP.md).

---

## FIR Artifacts (`artifact` module, always available)

`FirArtifact::parse(bytes)` is zero-copy and borrows the slice. It detects the container and decodes on demand into `&mut [f32]` (`decode_f32`), `&mut [i16]` (`decode_q15`) or `&mut [i32]` (`decode_q31`). Integer PCM is converted to fixed point with integer arithmetic only, so no FPU is needed.

| Container | Encodings | Channels |
| :--- | :--- | :--- |
| RIFF/WAVE (`PCM`, `IEEE_FLOAT`, `WAVE_FORMAT_EXTENSIBLE`) | PCM `u8`/`i16`/`i24`/`i32`, IEEE `f32`/`f64` | any (interleaved) |
| Raw binary (`.bin`) | `f32`/`f64` auto-detected; explicit `i16`/`i24`/`i32`/`f32`/`f64` via `parse_raw` | explicit (rePhase writes mono) |
| Text (`.txt`/`.csv`) | decimal point or decimal comma; `#`, `*`, `%` and header lines skipped | columns = channels |

Typed errors (`ArtifactError`) cover truncation, unsupported formats, non-finite samples and misalignment. rePhase's export dialog is not formally specified. Every listed format is tested against fixtures synthesized from the RIFF specification (`tests/artifact_formats.rs`); no real rePhase file was available for testing.

---

## Feature Flags

| Feature | Enables | Adds `std`? |
| :--- | :--- | :---: |
| `auto` *(default)* | `AutoConvolver`, `AutoStereoConvolver`, `AutoWorkspace`; implies `fixed-nonuniform` | No |
| `native-q15` | Forces the Q1.15 auto domain on FPU targets (RAM savings, host testing) | No |
| `native-q31` | Forces the Q1.31 auto domain on any target (24/32-bit audio without an FPU; takes precedence over `native-q15`) | No |
| `nonuniform` | `NonUniformConvolver`, `NonUniformPlan`, `NonUniformBudget`, `NonUniformScheduler` | No |
| `fixed-q15` | `FixedQ15DirectConvolver` and Q1.15 helpers | No |
| `fixed-fft` | BFP `rfft_q15_bfp_with_scratch` / `irfft_q15_bfp_with_scratch` (`N64..=N2048`, `1,026 B` ROM) and `FixedQ15PartitionedConvolver` | No |
| `fixed-nonuniform` | `FixedQ15NonUniformConvolver` | No |
| `fixed-q31` | `FixedQ31DirectConvolver` (FIR Type I folding), `FixedQ31PartitionedConvolver`, `FixedQ31NonUniformConvolver` on the 32-bit-mantissa BFP FFT core; `AutoElem for i32`, `AutoConvolver::process_i32` | No |
| `std` | Runtime CPU detection (`SimdBackend::detect()`) | Yes |
| `phastft` | Optional desktop `PhastFftBackend` (`phastft 0.4.1`, `N64..=N16384`) | Yes |

`--no-default-features` gives the minimal core (`Direct`, `Partitioned`, `Hybrid`, the FFT backends and `artifact`). Every feature except `std`/`phastft` is `no_std` with no `alloc`.

```toml
[dependencies]
rtconv-simd = { git = "https://github.com/SM-Acoustics/rtconv-simd", branch = "main" }
```

> **Planned work**: [`docs/ROADMAP.md`](./docs/ROADMAP.md) and issue [#13](https://github.com/SM-Acoustics/rtconv-simd/issues/13).

---

## Manual Engines

### Zero-latency hybrid (`f32` with tail spreading)

```rust
use rtconv_simd::{HybridConfig, HybridConvolver, SimdBackend, SimdFixedFftBackend};

let ir = [0.5f32, 0.25, -0.125, 0.0625];
let backend = SimdBackend::detect_or_baseline();
let fft = SimdFixedFftBackend::new(backend);
let cfg = HybridConfig { head_len: 64, tail_block_size: 64, max_ir_len: 128 };
let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
let mut workspace = vec![0.0f32; req.total_f32];
let mut conv =
    HybridConvolver::prepare_with_tail_spreading(&ir, cfg, &mut workspace, backend, fft, true)
        .unwrap();
let mut input = [0.0f32; 32];
input[0] = 1.0;
let mut output = [0.0f32; 32];
conv.process_block(&input, &mut output).unwrap();
assert!((output[0] - 0.5).abs() < 1e-6);
```

### Fixed-point NUPC from high-precision taps (FPU-less MCUs)

```rust
use rtconv_simd::{FixedQ15NonUniformConvolver, NonUniformPlan};

let ir_q31: Vec<i32> = (0..2048).map(|k| if k == 0 { i32::MAX / 2 } else { 1 << 16 }).collect();
let plan = NonUniformPlan::default_profile(ir_q31.len(), 32).unwrap();
let req = FixedQ15NonUniformConvolver::memory_requirements(&plan).unwrap();
let mut ws = vec![0i16; req.total_i16];
// Each FFT partition keeps its own block exponent (precise low-level tails).
let mut conv = FixedQ15NonUniformConvolver::prepare_q31(&ir_q31, plan, None, &mut ws).unwrap();
let mut out = [0i16; 32];
conv.process_block(&[1000i16; 32], &mut out).unwrap();
assert_eq!(conv.latency_samples(), 0);
```

More: [`examples/`](./examples) (`direct_fir`, `hybrid_convolution`, `nonuniform_convolution --features nonuniform`, `fixed_q15_direct --features fixed-q15`, `backend_selection --all-features`, `auto_from_artifact`).

---

## FFT Backends

| Backend | Feature | Precision | Supported `N` | SIMD transform | External scratch | ROM / Peak stack |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **`MicroFftBackend`** | core | `f32` | `64..=16384` | No (radix-2 + Hermitian inverse) | `N` `f32` (`256 B`–`64 KiB`) | `16,380 B` ROM / `< 256 B` stack |
| **`SimdFixedFftBackend`** | core | `f32` | `64..=16384` | Yes (`fearless_simd`; scalar on RISC-V) | `N` `f32` (`256 B`–`64 KiB`) | `16,388 B` ROM / `< 256 B` stack |
| **`PhastFftBackend`** | `phastft` (`std`) | `f32` | `64..=16384` | Yes (`phastft 0.4.1`) | `N` `f32` (`256 B`–`64 KiB`) | precomputed planner / `< 256 B` stack |
| **`fft_q15` (BFP radix-2)** | `fixed-fft` | Q1.15 mantissa + `i8` exp | `64..=2048` | No (FPU-free `i32` butterflies) | `2N` `i32` / `4N` `i16` (`in ws`) | `1,026 B` ROM / `< 512 B` stack (engines) |

---

## Benchmarks & Numerical Verification

### Timing (`HOST_ONLY`)

Measured at 48 kHz, 32-sample blocks (`666.67 µs` budget); full GUM Type A uncertainty (`k = 2`) and tail percentile tables in [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md).

- **Windows 11 / AMD Ryzen 3 PRO 2300U (`0.2.0` baseline, `N ≤ 256`)**: `Hybrid(H128/T128) SPREAD` at 16,383 taps worst-phase p99 = **5.20 µs (0.78 % of budget)**.
- **Linux x86-64 / Intel Xeon 2.30 GHz (PR-A/B/C `#14`–`#16`, `N64..=N16384`, `SimdFixedFftBackend`, `n = 4096`)**:
  - `65,536` taps (`H = 32`, time-distributed NUPC): `NUPC optimized (Strategy1)` achieves **`8.78 ± 0.30 µs`** (`p95 = 15.12 µs`, `p99 = 39.79 µs` = `5.97%` of budget) on SSE2 — beating `Hybrid(H512/T512) SPREAD` (`8.84 ± 0.20 µs`, `p95 = 17.59 µs`) — and **`9.75 ± 0.38 µs`** (`p99 = 48.02 µs`) on AVX2+FMA; at `16,384` taps AVX2+FMA, `NUPC Strategy1` achieves **`p99 = 15.25 µs`** (`2.29%` of budget, beating `Hybrid(H512)` `15.62 µs`).
  - Deterministic sliding-window op-trace (`tests/time_distributed_fft.rs`): `Strategy1` peak/mean op ratio stays **`1.21×–1.76×`** across `4k–64k` taps (`2.69×` lower peak ops than monolithic at `64k`), while `2 * B_s` bitmask rings save `−21` to `−37 KiB` on large `f32` plans and `−16 KiB` scratch + `−75%` delay rings on high-offset `Q1.15` plans.
  - Isolated `RFFT + IFFT` (`N = 128..=8192`): `SimdFixedFft(AVX2)` reaches up to **`1.28×`** over zero-stack `MicroFftBackend` (`PhastFftBackend` reaches **`3.25×`**).

### Numerical accuracy (`HOST_TESTED`)

Measured with `tests/fft_backends_suite.rs`, `tests/adversarial_reference.rs`, `tests/aes17_audio_metrics.rs`, and `tests/auto_convolver.rs`.

- **FFT `f64` DFT oracle (`N64..=N16384`, all `f32` backends)**: forward SNR **`132.5–143.0 dB`**, roundtrip SNR **`131.9–140.1 dB`** (`> 120 dB` gate).
- **`f32` engines** (all backends, IR lengths `1..=131,072`, random host fragmentation `1..3·B`): SNR **`> 100 dB`** against an independent `f64` oracle (`131.77 dB` on 131,072-tap NUPC up to `N16384`).
- **AES17 (`997 Hz` @ `48 kHz`)**:
  - `f32` (`N512..=N16384`): SNR/THD+N **`127.9–135.9 dB`** across `−60 / −1 / 0 dBFS`, DR **`193.4–201.8 dB`** at `−60 dBFS`, zero overshoot.
  - `Q1.15` (`N64..=N2048`): at `−60 dBFS`, input-block BFP preserves **`97.87–98.65 dB` DR** (`0.00 dB` loss vs Q1.15 ceiling). Since issue #23 the engines transform with 32-bit mantissas (`fft_q31` core, ≥ `155 dB` oracle SNR); at `−1 dBFS` a pure tone is now limited by the 16-bit BFP storage of the partition spectra (`81.89 dB` unity impulse at `N=64`, previously `76.72 dB`), and a 4,096-tap rePhase-style FIR on noise reaches `82.36 dB` against an `85.18 dB` output-rounding ceiling (previously `72.00 dB`).
- **`FixedQ15Partitioned` / `FixedQ15NonUniform`** (decaying 1,500-tap IR):

  | Input level | Partitioned | NonUniform | Q1.15 output-rounding ceiling |
  | :--- | ---: | ---: | ---: |
  | 0 dBFS | 68.75 dB | 69.58 dB | 70.38 dB |
  | −20 dBFS | 50.38 dB | 50.40 dB | 50.40 dB |
  | −40 dBFS | 30.53 dB | 30.53 dB | 30.53 dB |

- **4,096-tap linear-phase (rePhase-style) FIR, Q1.15 NUPC**: **`72.00 dB`** with `prepare_q31` per-partition coefficient exponents and convergent RNE rounding, vs. `64.56 dB` with Q1.15 taps. Reaching `> 90 dB` at large `N` in fixed point requires 32-bit BFP mantissas (#23).

---

## Real-Time Guarantees and Limits

- **Verified invariants (`HOST_ONLY`)**:
  - `#![no_std]` and `#![forbid(unsafe_code)]`.
  - 0 heap allocations or deallocations (standalone `harness = false` `GlobalAlloc` binary with a positive control), covering every `f32` (`N64..=N16384`) and Q1.15 (`N64..=N2048`) engine, artifact parsing, and `AutoConvolver`.
  - No locks, threads or I/O; bounded loops; typed `ConvError` / `ArtifactError` rejections; `< 512 B` peak stack across all engines.
  - **IEEE 754 subnormal guard.** Every `f32` engine flushes values below `simd_kernels::DENORMAL_GUARD = 2^-63` (≈ −379 dBFS) at its boundaries: inputs, coefficients, spectra, FDL slots, accumulators and outputs (`tests/denormal_guard.rs`, up to `N16384`).
    - Without the guard, on x86 without FTZ/DAZ, decaying or near-silent signals made `DirectConvolver` up to 80× slower and `HybridConvolver` up to 50× slower, and an f64 FIR with a deep tail made it 15× slower permanently.
    - With the guard, every case runs at the normal-signal cost (`benches/denormal_bench.rs`, [`docs/ROADMAP.md` §4](./docs/ROADMAP.md)).
- **Limits**:
  - Host benchmarks are not WCET certifications.
  - `NonUniformBudget` is an operation-count admission model, not a timing proof.
  - `FixedQ15NonUniformConvolver` now keeps its per-stage `i64` accumulators and FFT scratch inside the caller `&mut [i16]` workspace, keeping `instance_bytes` compact (`< 4 KiB`, still included in `NonUniformBudget` admission).
  - See [`docs/REALTIME_CONTRACT.md`](./docs/REALTIME_CONTRACT.md) and [`docs/MEMORY_MODEL.md`](./docs/MEMORY_MODEL.md).

---

## Platform Support & Verification Status

| Target | Build | Tests | Hardware | Numeric path |
| :--- | :--- | :--- | :---: | :--- |
| **Linux x86-64** | `CI_VERIFIED` (stable + MSRV 1.89.0) | `CI_VERIFIED` | `CI_HOST_ONLY` | SSE2/SSE4.2/AVX2+FMA + Q1.15/BFP |
| **Windows x86-64** (MSVC & GNU) | `CI_VERIFIED` + `HOST_TESTED` | `CI_VERIFIED` + `HOST_TESTED` | `HOST_ONLY` | as above |
| **macOS Apple Silicon** | `CI_VERIFIED` | `CI_VERIFIED` | `CI_HOST_ONLY` | NEON + Q1.15/BFP |
| **ESP32-C6** (`riscv32imac-unknown-none-elf`) | `CROSS_COMPILE_VERIFIED` (core, all no_std features, `hardware/esp32c6` harness) | host mirror only | `NO` | Q1.15/BFP auto domain |
| **ESP32-P4** (`riscv32imafc-unknown-none-elf`) | `CROSS_COMPILE_VERIFIED` (core, all no_std features) | `OPEN` | `NO` | `f32` scalar (no PIE/RVV) |
| **Cortex-M4F/M7** (`thumbv7em-none-eabihf`) | `CROSS_COMPILE_VERIFIED` | `OPEN` | `NO` | `f32` scalar |
| **Cortex-M0** (`thumbv6m-none-eabi`) | `CROSS_COMPILE_VERIFIED` | `OPEN` | `NO` | Q1.15/BFP |
| **ARM64 bare-metal / WASM** | `CROSS_COMPILE_VERIFIED` | `OPEN` | `NO` | NEON / SIMD128 or fallback |

See [`docs/PORTABILITY.md`](./docs/PORTABILITY.md) and [`hardware/esp32c6/README.md`](./hardware/esp32c6/README.md).

---

## Documentation

- **Guides**: [`ARCHITECTURE`](./docs/ARCHITECTURE.md) · [`MEMORY_MODEL`](./docs/MEMORY_MODEL.md) · [`SIMD_BACKENDS`](./docs/SIMD_BACKENDS.md) · [`REALTIME_CONTRACT`](./docs/REALTIME_CONTRACT.md) · [`BENCHMARKS`](./docs/BENCHMARKS.md) · [`PORTABILITY`](./docs/PORTABILITY.md)
- **Review**: [`docs/REVIEW-2026-11-ADVERSARIAL.md`](./docs/REVIEW-2026-11-ADVERSARIAL.md) (findings, fixes, literature, residual risks)
- **History**: [`ROADMAP`](./docs/ROADMAP.md) · [`CHANGELOG`](./CHANGELOG.md) · work-order reports `docs/WO-RTCONV-004*.md`
- **Contributing**: [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) · [`SECURITY.md`](./SECURITY.md)

## License

Licensed under the GNU General Public License v3.0 or later ([`GPL-3.0-or-later`](./LICENSE)).

Commercial dual-licensing is available upon request from the copyright holder (SM Acoustics) for proprietary or closed-source integration.
