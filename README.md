# RTConv SIMD (`rtconv-simd`)

**Zero-latency, zero-allocation FIR convolution in pure Rust, from AVX2/NEON desktops to FPU-less RISC-V microcontrollers. Give it a FIR file and it picks the engine.**

`#![no_std]` · `#![forbid(unsafe_code)]` · no heap (`alloc` is never used) · MSRV `1.89.0` · `MIT OR Apache-2.0`

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
| `riscv32imac-unknown-none-elf` | **ESP32-C6**, ESP32-H2 | Q1.15 / BFP (no FPU) | `FixedQ15Direct` (≤ 128 taps) or `FixedQ15NonUniform` |
| `riscv32imafc-unknown-none-elf` | **ESP32-P4** | `f32` (hardware FPU, scalar) | `Direct` or `Hybrid` (`B` = 32/64/128) |
| `thumbv6m-none-eabi`, `thumbv7m-none-eabi` | Cortex-M0/M3 | Q1.15 / BFP | as ESP32-C6 |
| `thumbv7em-none-eabihf` | Cortex-M4F/M7 | `f32` | as ESP32-P4 |
| x86-64 / AArch64 / WASM | desktop, mobile, web | `f32` + SIMD | as ESP32-P4 |

The FPU is detected by `build.rs` from the target ISA string. Stable Rust does not expose the RISC-V `F` extension as a `cfg`. The `native-q15` feature forces the fixed-point domain on any target.

---

## Convolution Engines

All core engines are mono; use one instance per channel, or the `auto` stereo wrapper.

| Engine | Feature | Numeric format | Latency | Best for |
| :--- | :--- | :--- | :---: | :--- |
| **`AutoConvolver` / `AutoStereoConvolver`** | `auto` (default) | `f32` or Q1.15 (by target) | `0` | Loading FIR artifacts with zero configuration |
| **`DirectConvolver`** | core | `f32` (SIMD / scalar) | `0` | Short FIRs (≲ 170 taps by the cost model), crossovers |
| **`PartitionedConvolver`** | core | `f32` (uniform 50 % overlap-save FDL) | `0`¹ | Block-aligned FFT convolution |
| **`HybridConvolver`** | core | `f32` (direct head + FDL tail, `SPREAD`) | `0` | Long IRs on desktop and FPU MCUs |
| **`NonUniformConvolver`** | `nonuniform` | `f32` (up to 8 causal stages) | `0` | Multi-stage schedules with budget admission |
| **`FixedQ15DirectConvolver`** | `fixed-q15` | Q1.15 (`i64` accumulator) | `0` | FPU-less MCUs, ≤ 1 LSB error, half the RAM of `f32` |
| **`FixedQ15PartitionedConvolver`** | `fixed-fft` | Q1.15 + BFP FFT/FDL | `0`¹ | Medium/long FIRs on FPU-less MCUs |
| **`FixedQ15NonUniformConvolver`** | `fixed-nonuniform` | Q1.15 + BFP NUPC | `0` | Zero-latency long FIRs on FPU-less MCUs |

¹ The output for an input block is produced in the same call. A host that delivers fewer than `B` samples per callback adds `B` samples of framing delay, which is why the hybrid and NUPC engines exist.

> **FFT ceiling**: every FFT backend supports `N ∈ {64, 128, 256}` (`B ∈ {32, 64, 128}`). Larger sizes are the next milestone ([`docs/ROADMAP.md`](./docs/ROADMAP.md)).

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
| `nonuniform` | `NonUniformConvolver`, `NonUniformPlan`, `NonUniformBudget`, `NonUniformScheduler` | No |
| `fixed-q15` | `FixedQ15DirectConvolver` and Q1.15 helpers | No |
| `fixed-fft` | BFP `rfft_q15_bfp` / `irfft_q15_bfp` (512 B twiddle ROM) and `FixedQ15PartitionedConvolver` | No |
| `fixed-nonuniform` | `FixedQ15NonUniformConvolver` | No |
| `std` | Runtime CPU detection (`SimdBackend::detect()`) | Yes |
| `phastft` | Optional desktop `PhastFftBackend` (`phastft 0.4.1`) | Yes |

`--no-default-features` gives the minimal core (`Direct`, `Partitioned`, `Hybrid`, the FFT backends and `artifact`). Every feature except `std`/`phastft` is `no_std` with no `alloc`.

```toml
[dependencies]
rtconv-simd = { git = "https://github.com/SM-Acoustics/rtconv-simd", branch = "main" }
```

> **Release status**: the crate version on `main` is **`0.2.0`** (see [`CHANGELOG.md`](./CHANGELOG.md)). The latest GitHub release is `v0.1.0`. Tagging `v0.2.0` and publishing to crates.io are maintainer actions. Planned work: [`docs/ROADMAP.md`](./docs/ROADMAP.md) and issue [#13](https://github.com/SM-Acoustics/rtconv-simd/issues/13).

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

| Backend | Feature | Precision | SIMD transform | External scratch (`N = 64/128/256`) | Peak stack |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`MicroFftBackend`** | core | `f32` | No (radix-2) | `0 B` | up to ~2.1 KiB (`N = 256`) |
| **`SimdFixedFftBackend`** | core | `f32` | Yes (`fearless_simd`; scalar on RISC-V) | `256 / 512 / 1024 B` | `< 256 B` |
| **`PhastFftBackend`** | `phastft` (`std`) | `f32` | Yes (`phastft 0.4.1`) | `256 / 512 / 1024 B` | `< 256 B` |
| **`fft_q15` (BFP radix-2)** | `fixed-fft` | Q1.15 mantissa + `i8` exponent | No (FPU-free `i32` butterflies) | caller `&mut [i16]` (512 B ROM) | ~2 KiB per transform (+ 2 KiB `i64` accumulator in the fixed engines) |

---

## Benchmarks & Numerical Verification

### Timing (`HOST_ONLY`)

Measured on an AMD Ryzen 3 PRO 2300U (Windows 11, 48 kHz, 32-sample blocks, 666.67 µs budget). See [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md).

- **SIMD FFT** (`RFFT + IFFT`, `SimdFixedFftBackend` vs `MicroFftBackend`): 2.08× at `N = 64` (SSE2), 2.85× at `N = 128` (AVX2), 2.51× at `N = 256` (AVX2).
- **16,383 taps**: `Hybrid(H128/T128) SPREAD` worst-phase p99 = **5.20 µs (0.78 % of budget)**.

### Numerical accuracy (`HOST_TESTED`)

Measured with `tests/adversarial_reference.rs` and `tests/auto_convolver.rs`.

- **`f32` engines** (all backends, IR lengths 1–4097, random host fragmentation 1–3·`B`): SNR > 100 dB against an independent `f64` oracle.
- **`FixedQ15Partitioned` / `FixedQ15NonUniform`** (decaying 1,500-tap IR):

  | Input level | Partitioned | NonUniform | Q1.15 output-rounding ceiling |
  | :--- | ---: | ---: | ---: |
  | 0 dBFS | 68.7 dB | 69.4 dB | 70.4 dB |
  | −20 dBFS | 50.4 dB | 50.4 dB | 50.4 dB |
  | −40 dBFS | 30.5 dB | 30.5 dB | 30.5 dB |

  Before the BFP fix in the [review](./docs/REVIEW-2026-11-ADVERSARIAL.md), partitioned measured 47.8 / 46.0 / 30.5 dB and NonUniform 47.7 / 43.0 / 25.8 dB.
- **4,096-tap linear-phase (rePhase-style) FIR, Q1.15 NUPC**: 70.4 dB with `prepare_q31` per-partition coefficient exponents, vs. 64.2 dB with Q1.15 taps (62.6 dB before the review). The Q1.15 FFT path tops out at about 70 dB (16-bit mantissas).

---

## Real-Time Guarantees and Limits

- **Verified invariants (`HOST_ONLY`)**:
  - `#![no_std]` and `#![forbid(unsafe_code)]`.
  - 0 heap allocations or deallocations (standalone `harness = false` `GlobalAlloc` binary with a positive control), covering every `f32` and Q1.15 engine, artifact parsing, and `AutoConvolver`.
  - No locks, threads or I/O; bounded loops; typed `ConvError` / `ArtifactError` rejections.
  - **IEEE 754 subnormal guard.** Every `f32` engine flushes values below `simd_kernels::DENORMAL_GUARD = 2^-63` (≈ −379 dBFS) at its boundaries: inputs, coefficients, spectra, FDL slots, accumulators and outputs. Every non-zero product therefore stays a normal number.
    - Without the guard, on x86 without FTZ/DAZ, decaying or near-silent signals made `DirectConvolver` up to 80× slower and `HybridConvolver` up to 50× slower, and an f64 FIR with a deep tail made it 15× slower permanently.
    - With the guard, every case runs at the normal-signal cost (`benches/denormal_bench.rs`, [`docs/ROADMAP.md` §4](./docs/ROADMAP.md)).
- **Limits**:
  - Host benchmarks are not WCET certifications.
  - `NonUniformBudget` is an operation-count admission model, not a timing proof.
  - `FixedQ15NonUniformConvolver` embeds about 17 KiB of `i64` accumulators in the struct itself. This is reported as `instance_bytes`; keep the struct in a `static` on MCUs.
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

Dual-licensed under [`MIT`](./LICENSE-MIT) or [`Apache-2.0`](./LICENSE-APACHE), at your option.
