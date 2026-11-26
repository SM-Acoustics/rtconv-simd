# RTConv SIMD (`rtconv-simd`)

**Deterministic, zero-allocation FIR convolution in pure Rust — from AVX2/NEON desktop audio engines to FPU-less RISC-V microcontrollers.**

`#![no_std]` · `#![forbid(unsafe_code)]` · `MSRV 1.89.0` · `Zero Heap` · `GPL-3.0-or-later`

`rtconv-simd` (library crate `rtconv_simd`) is a real-time FIR convolution suite with caller-managed workspaces, pluggable SIMD FFT backends, multi-stage non-uniform partitioning (NUPC), and a complete Q1.15 / Block Floating Point (BFP) fixed-point stack.

---

## Why RTConv SIMD?

- **Strict Real-Time Core**: `#![no_std]`, `#![forbid(unsafe_code)]`, `default = []` (zero `std`, zero `alloc`). Every working buffer is supplied by the caller (`&mut [f32]` or `&mut [i16]`) using checked arithmetic — verified to perform **0 heap allocations** across `prepare()`, `process_block()`, and `reset()`.
- **Zero-Latency Long IRs**: Combine a direct FIR head with a time-distributed FFT tail (`HybridConvolver` with `SPREAD`) or a multi-stage causal schedule (`NonUniformConvolver`) for **0-sample algorithmic latency** regardless of host callback block size.
- **Desktop SIMD + FPU-Less MCU Ready**: Accelerates `f32` via `fearless_simd 1.2.0` (SSE2, SSE4.2, AVX2+FMA, NEON, WASM SIMD128) and ships a full **Q1.15 / Block Floating Point** stack (`direct`, `FFT/FDL`, and `NUPC`) that cuts workspace RAM by **~50%** without floating-point emulation.
- **Phase-Agnostic & Bit-Honest**: Processes any finite real causal FIR impulse response verbatim — preserving exact tap order, gain, phase, and group delay with zero hidden normalization or windowing.
- **Evidence-Based Claims**: Explicitly separates `HOST_TESTED`, `CI_VERIFIED`, `CROSS_COMPILE_VERIFIED`, and `DEVICE_VERIFIED` — no synthetic cycle counts, no unverified WCET promises.

---

## Convolution Engines

All engines are mono (run independent instances for stereo/multichannel) and enforce caller-owned workspace sizing:

| Engine | Feature Gate | Numeric Format | Latency | Best For |
| :--- | :--- | :--- | :---: | :--- |
| **`DirectConvolver`** | default (`no_std`) | `f32` (SIMD / scalar) | `0` | Short FIRs (32–256 taps), crossovers, speaker/cab heads |
| **`PartitionedConvolver`** | default (`no_std`) | `f32` (50% overlap-save FDL) | `B` (`32/64/128`) | Uniform block-aligned FFT convolution |
| **`HybridConvolver`** | default (`no_std`) | `f32` (Direct head + FDL tail) | `0` | **Fastest desktop long IRs today** (`0.78%` p99 budget @ 16k taps with `SPREAD`) |
| **`NonUniformConvolver`** | `nonuniform` | `f32` (Multi-stage NUPC, up to 8 stages) | `0` | Multi-stage causal scheduling (`D_s >= B_s`), `SPREAD`, budget admission control |
| **`FixedQ15DirectConvolver`** | `fixed-q15` | `Q1.15` (`i16`, `i64` acc) | `0` | FPU-less MCUs (RV32IMAC, Cortex-M), 50% less RAM, `<= 1 LSB` error |
| **`FixedQ15PartitionedConvolver`** | `fixed-fft` | `Q1.15` + Block Floating Point FFT/FDL | `B` (`32/64/128`) | Medium/long FIRs on FPU-less MCUs without fixed-scale FFT SNR collapse |
| **`FixedQ15NonUniformConvolver`** | `fixed-nonuniform` | `Q1.15` + BFP Multi-stage NUPC | `0` | Zero-latency multi-stage convolution on embedded targets (reuses `NonUniformPlan` & `NonUniformScheduler`) |

> **FFT ceiling note**: All FFT backends (`MicroFftBackend`, `SimdFixedFftBackend`, `PhastFftBackend`, and `fft_q15`) currently support `N ∈ {64, 128, 256}` (`block_size ∈ {32, 64, 128}`). Adding `N512`/`N1024`/`N2048` is the next milestone in [`docs/ROADMAP.md`](./docs/ROADMAP.md) so multi-stage NUPC surpasses `Hybrid SPREAD` in throughput on very long room IRs (`>= 16k` taps).

---

## Feature Flags & Installation

> **Release status**: `v0.1.0` is published as a GitHub Release (`crates.io` publication pending). The `nonuniform`, `fixed-q15`, `fixed-fft`, and `fixed-nonuniform` engines are live on `main` targeting `0.2.0`.

```toml
[dependencies]
# Minimal embedded / no_std core (Direct, Partitioned, Hybrid, MicroFFT, SimdFixedFFT)
rtconv-simd = { git = "https://github.com/SM-Acoustics/rtconv-simd", branch = "main" }

# Desktop with runtime CPU detection (`SimdBackend::detect()`) + multi-stage NUPC
# rtconv-simd = { git = "https://github.com/SM-Acoustics/rtconv-simd", features = ["std", "nonuniform"] }

# FPU-less embedded MCU (no_std, zero alloc, full Q1.15/BFP multi-stage NUPC)
# rtconv-simd = { git = "https://github.com/SM-Acoustics/rtconv-simd", default-features = false, features = ["fixed-nonuniform"] }
```

| Feature | Enables | Adds `std`? |
| :--- | :--- | :---: |
| *(default)* | `DirectConvolver`, `PartitionedConvolver`, `HybridConvolver`, `MicroFftBackend`, `SimdFixedFftBackend` | No |
| `std` | Runtime CPU feature detection (`SimdBackend::detect()`) via `fearless_simd/std` | Yes |
| `phastft` | Optional desktop `PhastFftBackend` (`phastft 0.4.1`) | Yes |
| `nonuniform` | `NonUniformConvolver`, `NonUniformPlan`, `NonUniformBudget`, sample-time `NonUniformScheduler` | No |
| `fixed-q15` | `FixedQ15DirectConvolver` + Q1.15 math & saturation telemetry | No |
| `fixed-fft` | Q1.15/BFP FFT (`rfft_q15_bfp` / `irfft_q15_bfp`, `512 B` twiddle ROM) + `FixedQ15PartitionedConvolver` | No |
| `fixed-nonuniform` | `FixedQ15NonUniformConvolver` (reuses `nonuniform` + `fixed-fft`) | No |

---

## Quick Start

### 1. Zero-Latency Hybrid Convolution (`f32` + Tail Spreading)

```rust
use rtconv_simd::{HybridConfig, HybridConvolver, SimdBackend, SimdFixedFftBackend};

let ir = [0.5f32, 0.25, -0.125, 0.0625];
let backend = SimdBackend::detect_or_baseline();
let fft = SimdFixedFftBackend::new(backend);
let cfg = HybridConfig {
    head_len: 64,
    tail_block_size: 64,
    max_ir_len: 128,
};

let req = HybridConvolver::memory_requirements(cfg, &fft).unwrap();
let mut workspace = [0.0f32; 2048];
let mut conv = HybridConvolver::prepare_with_tail_spreading(
    &ir,
    cfg,
    &mut workspace[..req.total_f32],
    backend,
    fft,
    true,
)
.unwrap();

let mut input = [0.0f32; 32];
input[0] = 1.0;
let mut output = [0.0f32; 32];
conv.process_block(&input, &mut output).unwrap();
assert!((output[0] - 0.5).abs() < 1e-6);
```

### 2. Multi-Stage Non-Uniform Partitioning (`--features nonuniform`)

```rust
#[cfg(feature = "nonuniform")]
fn run_nupc() {
    use rtconv_simd::{
        NonUniformBudget, NonUniformConvolver, NonUniformPlan, SimdBackend, SimdFixedFftBackend,
    };

    let ir = [1.0f32; 256];
    // Default causal profile clipped to ir.len():
    // Direct(0..32) + FDL(B=32, 32..256) [+ FDL(B=64, 256..1024) + FDL(B=128, 1024..)]
    let plan = NonUniformPlan::default_profile(ir.len(), 32).unwrap();
    let backend = SimdBackend::detect_or_baseline();
    let fft = SimdFixedFftBackend::new(backend);
    let req = NonUniformConvolver::memory_requirements(&plan, &fft).unwrap();

    let budget = NonUniformBudget {
        sample_rate_hz: 48_000,
        max_frames_per_callback: 32,
        deadline_us: 666,
        max_memory_bytes: req.total_bytes * 2,
        max_ops_per_callback: 0,
        safety_margin_percent: 20,
    };

    let mut ws = vec![0.0f32; req.total_f32];
    let mut conv = NonUniformConvolver::prepare_with_budget(
        &ir,
        plan,
        Some(budget),
        &mut ws,
        backend,
        fft,
    )
    .unwrap();

    let mut input = [0.0f32; 32];
    input[0] = 1.0;
    let mut output = [0.0f32; 32];
    conv.process_block(&input, &mut output).unwrap();
    assert_eq!(conv.capabilities().latency_samples, 0);
    assert_eq!(conv.deadline_misses(), 0);
}
```

### 3. Fixed-Point Q1.15 Direct FIR (`--features fixed-q15`, FPU-Free)

```rust
#[cfg(feature = "fixed-q15")]
fn run_q15() {
    use rtconv_simd::fixed::f32_to_q15;
    use rtconv_simd::{FixedQ15DirectConfig, FixedQ15DirectConvolver};

    let ir = [f32_to_q15(0.5), f32_to_q15(-0.25), f32_to_q15(0.125)];
    let cfg = FixedQ15DirectConfig { max_ir_len: 32 };
    let req = FixedQ15DirectConvolver::memory_requirements(cfg).unwrap();
    let mut ws = [0i16; 64];

    let mut conv = FixedQ15DirectConvolver::prepare(&ir, cfg, &mut ws[..req.total_i16]).unwrap();
    let mut input = [0i16; 32];
    input[0] = f32_to_q15(0.5);
    let mut output = [0i16; 32];
    conv.process_block(&input, &mut output).unwrap();
    assert_eq!(conv.saturation_count(), 0);
}
```

---

## FFT Backends

| Backend | Feature | Precision | SIMD FFT | External Scratch (`N = 64/128/256`) | Peak Stack |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`MicroFftBackend`** | default (`no_std`) | `f32` | No (scalar radix-2) | `0 B` | ~2.1 KiB (`N=256`) |
| **`SimdFixedFftBackend`** | default (`no_std`) | `f32` | Yes (`fearless_simd 1.2.0`) | `256 / 512 / 1024 B` | `< 256 B` |
| **`PhastFftBackend`** | `phastft` (`std`) | `f32` | Yes (`phastft 0.4.1`) | `256 / 512 / 1024 B` | `< 256 B` |
| **`fft_q15` (BFP Radix-2)** | `fixed-fft` (`no_std`) | `Q1.15` + `i8` exp | No (FPU-free `i32` butterflies) | caller `&mut [i16]` (`512 B` ROM) | minimal |

See [`docs/SIMD_BACKENDS.md`](./docs/SIMD_BACKENDS.md) and [`docs/WO-RTCONV-004D-FIXED-FFT.md`](./docs/WO-RTCONV-004D-FIXED-FFT.md).

---

## Benchmarks & Numerical Verification

Empirical host measurements on **AMD Ryzen 3 PRO 2300U** (Zen 1, Windows 11 Pro, `rustc 1.100.0`, `--release`, `48 kHz` / `32`-sample block = **`666.67 µs` budget**, `HOST_ONLY`):

- **SIMD FFT Speedup (`RFFT + IFFT`, `SimdFixedFftBackend` vs `MicroFftBackend`)**:
  - `N=64`: **343.8 ns** (`2.08x`, SSE2)
  - `N=128`: **553.1 ns** (`2.85x`, AVX2)
  - `N=256`: **1,225.0 ns** (`2.51x`, AVX2)
- **Long IR (`16,383` taps, 32-sample callbacks)**:
  - `Hybrid(H128/T128)` + `SimdFixedFft(SSE2)` + `SPREAD`: **`5.20 µs` worst-phase p99 (`0.78%` of budget)**, down from `30.10 µs` (`4.52%`) with `MicroFft` burst.
  - `NonUniform(D32/32/64/128)`: `mean = 8.91 µs`, `p99 = 20.50 µs` (`3.08%`), `deadline_misses = 0` (`Hybrid SPREAD` remains faster while FFT max size is `N256`).
- **Fixed-Point Q1.15 / BFP Accuracy vs `f64` Oracle**:
  - `FixedQ15DirectConvolver` (512 taps): `max_abs <= 1 LSB (3.05e-5)`, `RMS = 8.911e-6`, **`SNR = 69.55 dB`**, **50% RAM** vs `f32` (`128..2,048 B` for `32..512` taps).
  - `rfft_q15_bfp` / `irfft_q15_bfp` (`N64/N128/N256`): forward **`SNR > 62 dB`**, roundtrip **`SNR > 60 dB`**, `0` saturations.
  - `FixedQ15PartitionedConvolver` (500 taps): **`SNR = 51.36 dB`**, `max_abs = 1.899e-4`, `0` saturations.
  - `FixedQ15NonUniformConvolver` (1,350 taps, 4 stages, fragmented blocks): **`SNR = 44.19 dB`** (-36 dBFS output), `deadline_misses = 0`, `saturation_count = 0`.

Full tables and commands live in [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) and [`docs/MEMORY_MODEL.md`](./docs/MEMORY_MODEL.md).

---

## Real-Time Guarantees and Limits

- **Verified Invariants (`PROVEN`, `HOST_ONLY`)**: `#![no_std]`, `#![forbid(unsafe_code)]`, **0 heap allocations/deallocations** verified via a standalone `harness = false` `GlobalAlloc` test binary (`tests/no_alloc_rt.rs`) with a deliberate positive allocation control across all `f32` and `Q1.15` engines, zero locks/threads/I/O, bounded loops, and typed `ConvError` rejections.
- **Limits**: Host benchmarks on general-purpose OSes are **not** formal WCET certifications. Hard real-time qualification requires target-specific static timing analysis and hardware-in-the-loop profiling. See [`docs/REALTIME_CONTRACT.md`](./docs/REALTIME_CONTRACT.md).

---

## Platform Support & Verification Status

| Target / Architecture | Build & Cross-Compile | Tests (`CI` / `HOST`) | Hardware (`DEVICE_VERIFIED`) | SIMD / Numeric Path |
| :--- | :--- | :--- | :---: | :--- |
| **Linux x86-64** (`ubuntu-latest`) | `CI_VERIFIED` (`stable` + MSRV `1.89.0`) | `CI_VERIFIED` | `CI_HOST_ONLY` | SSE2, SSE4.2, AVX2+FMA + Q1.15/BFP |
| **Windows x86-64** (`MSVC` & `GNU`) | `CI_VERIFIED` + `HOST_TESTED` | `CI_VERIFIED` + `HOST_TESTED` | `HOST_ONLY` | SSE2, SSE4.2, AVX2+FMA + Q1.15/BFP |
| **macOS Apple Silicon** (`macos-latest`) | `CI_VERIFIED` | `CI_VERIFIED` (incl. `no_alloc_rt`) | `CI_HOST_ONLY` | NEON via `fearless_simd` + Q1.15/BFP |
| **ARM64 bare-metal** (`aarch64-unknown-none`) | `CROSS_COMPILE_VERIFIED` (`no_std`) | `OPEN` | `NO` | NEON via `fearless_simd` |
| **WebAssembly** (`wasm32-unknown-unknown`) | `CROSS_COMPILE_VERIFIED` (`no_std`) | `OPEN` | `NO` | SIMD128 or fallback |
| **Cortex-M4F/M7F** (`thumbv7em-none-eabihf`) | `CROSS_COMPILE_VERIFIED` (`no_std`) | `OPEN` | `NO` | Scalar fallback / Q1.15/BFP |
| **RISC-V / ESP32-C6** (`riscv32imac-unknown-none-elf`) | `CROSS_COMPILE_VERIFIED` (core + `hardware/esp32c6/`) | `OPEN` | `NO` (harness ready) | Scalar / FPU-free Q1.15/BFP (no RVV claimed) |

See [`docs/PORTABILITY.md`](./docs/PORTABILITY.md) and [`hardware/esp32c6/README.md`](./hardware/esp32c6/README.md).

---

## Examples

```bash
cargo run --release --example direct_fir
cargo run --release --example hybrid_convolution
cargo run --release --example backend_selection --all-features
cargo run --release --example multiple_instances
cargo run --release --example all_engines
cargo run --release --example no_std_static
cargo run --release --example nonuniform_convolution --features nonuniform
cargo run --release --example fixed_q15_direct --features fixed-q15
```

---

## Documentation & Engineering Reports

- **Core Guides**: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) · [`docs/MEMORY_MODEL.md`](./docs/MEMORY_MODEL.md) · [`docs/SIMD_BACKENDS.md`](./docs/SIMD_BACKENDS.md) · [`docs/REALTIME_CONTRACT.md`](./docs/REALTIME_CONTRACT.md) · [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) · [`docs/PORTABILITY.md`](./docs/PORTABILITY.md)
- **Roadmap & Changelog**: [`docs/ROADMAP.md`](./docs/ROADMAP.md) · [`CHANGELOG.md`](./CHANGELOG.md)
- **Engineering Audit (`WO-RTCONV-004`)**: [`docs/WO-RTCONV-004-FINAL-REPORT.md`](./docs/WO-RTCONV-004-FINAL-REPORT.md) (CI/macOS, NUPC `f32`, Q1.15 Direct, Q1.15/BFP FFT, Fixed NUPC, and ESP32-C6 harness)
- **Contributing**: [`CONTRIBUTING.md`](./CONTRIBUTING.md) · [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) · [`SECURITY.md`](./SECURITY.md)

---

## License

Licensed under the GNU General Public License v3.0 or later ([`GPL-3.0-or-later`](./LICENSE)).

Commercial dual-licensing is available upon request from the copyright holder (SM Acoustics) for proprietary or closed-source integration.

*Historical note:* Release tag `v0.1.0` remains available under its original `MIT OR Apache-2.0` terms.
