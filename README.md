# RTConv SIMD (`rtconv-simd`)

Pure Rust, `#![no_std]`, `#![forbid(unsafe_code)]` FIR convolution with SIMD FFT, caller-managed memory, and predictable real-time DSP workloads.

## 1. Overview

`rtconv-simd` (library name `rtconv_simd`) provides time-domain direct FIR convolution, uniform partitioned frequency-domain delay line (FDL) convolution, and zero-latency hybrid convolution for embedded and desktop audio DSP. All audio-path working memory is supplied by the caller via `&mut [f32]` slices with checked size calculation, avoiding heap allocation in both initialization (`prepare`) and processing (`process_sample`, `process_block`) for the core `no_std` engines.

## 2. Features

- `#![no_std]` and `#![forbid(unsafe_code)]` in the core library (`default = []`, zero `std` and zero `alloc`).
- Three independent mono `f32` engines (multi-channel via independent instances):
  - `DirectConvolver`: circular-buffer time-domain FIR (`0`-sample latency).
  - `PartitionedConvolver`: 50% overlap-save uniform FDL (`N = 64, 128, 256`).
  - `HybridConvolver`: direct FIR head + partitioned FFT tail (`0`-sample algorithmic latency) with optional time-distributed FDL tail spreading (`set_tail_spreading(true)`).
- Phase-agnostic: preserves the exact coefficients, order, gain, phase, and group delay of any finite real causal FIR impulse response.
- Pluggable `FftBackend` trait with three implementations: `MicroFftBackend` (`no_std` scalar baseline), `SimdFixedFftBackend` (`no_std` SIMD using [`fearless_simd 1.2.0`](https://docs.rs/fearless_simd/1.2.0)), and optional desktop `PhastFftBackend` ([`phastft 0.4.1`](https://docs.rs/phastft), behind `--features phastft`).

## 3. Installation

```toml
[dependencies]
# Embedded / no_std default (zero std, zero alloc)
rtconv-simd = "0.1.0"

# Desktop with runtime CPU feature detection (`SimdBackend::detect()`)
# rtconv-simd = { version = "0.1.0", features = ["std"] }

# Desktop with optional PhastFT SIMD FFT backend
# rtconv-simd = { version = "0.1.0", features = ["phastft"] }
```

## 4. Quick Start

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

## 5. Direct, Partitioned, and Hybrid Engines

- **`DirectConvolver`**: Reverse-indexed circular buffer enables unit-stride forward SIMD dot products with a single `dispatch!` per block and zero per-sample history copies. Best for short FIRs (32–256 taps).
- **`PartitionedConvolver`**: Uniform 50% overlap-save FDL (`block_size ∈ {32, 64, 128}`, `FFT ∈ {64, 128, 256}`). Partition spectra are precomputed during `prepare()` into split-complex (SoA) planes (`N/2 + 1` bins) for contiguous SIMD CMAC.
- **`HybridConvolver`**: Runs `DirectConvolver` on `h[0..head_len]` and `PartitionedConvolver` on `h[head_len..]` with `head_len == tail_block_size`. Because the tail depends strictly on samples from preceding blocks, algorithmic latency is **0 samples** for any host block size (e.g., 32-sample host blocks with 128-sample tail partitions). Enabling `set_tail_spreading(true)` distributes past-partition CMACs across pre-completion callbacks without changing temporal alignment or latency.

## 6. FFT Backends

| Backend | Crate Feature | SIMD FFT | External Scratch (`N = 64/128/256`) | Peak Stack |
| :--- | :--- | :--- | :--- | :--- |
| `MicroFftBackend` | default (`no_std`) | No (scalar radix-2) | `0 B` | ~2.1 KiB (`N=256`) |
| `SimdFixedFftBackend` | default (`no_std`) | Yes (`fearless_simd 1.2.0`) | `256 / 512 / 1024 B` | `< 256 B` |
| `PhastFftBackend` | `phastft` (desktop) | Yes (`phastft 0.4.1`) | `256 / 512 / 1024 B` | `< 256 B` |

See [`docs/SIMD_BACKENDS.md`](./docs/SIMD_BACKENDS.md) for architecture and disassembly details.

## 7. Memory Requirements

All workspace sizes are computed via checked arithmetic (`memory_requirements`). Example caller-owned workspace footprints (48 kHz, `SimdFixedFftBackend`):

| Engine / Config | IR = 256 taps | IR = 2,048 taps | IR = 16,383 taps |
| :--- | ---: | ---: | ---: |
| `DirectConvolver` | 2.00 KiB | 16.00 KiB | 127.99 KiB |
| `PartitionedConvolver` (`B=64, FFT=128`) | 5.01 KiB | 34.26 KiB | 261.76 KiB |
| `HybridConvolver` (`H128/T128, FFT=256`) | 4.76 KiB | 35.74 KiB | 261.49 KiB |

See [`docs/MEMORY_MODEL.md`](./docs/MEMORY_MODEL.md) for exact formulas and buffer layouts.

## 8. Real-Time Guarantees and Limitations

- **Verified design invariants (`PROVEN`, `HOST_ONLY`)**: `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap allocations/deallocations in audio callbacks across all three FFT backends, zero locks/threads/I/O, bounded loop counts, and typed errors on invalid configuration or mid-cycle mode transitions.
- **Limitations**: Empirical host benchmarks on general-purpose operating systems are **not** formal WCET certifications. Hard real-time qualification requires target-specific static timing analysis and hardware-in-the-loop testing. See [`docs/REALTIME_CONTRACT.md`](./docs/REALTIME_CONTRACT.md).

## 9. Platform Support

| Target / Architecture | Compilation | Execution & Tests | SIMD Status |
| :--- | :--- | :--- | :--- |
| Windows x86-64 (`x86_64-pc-windows-gnu`) | `PROVEN` | `PROVEN` (`HOST_ONLY`) | SSE2, SSE4.2, AVX2+FMA verified |
| Linux x86-64 | CI configured | CI configured | SSE2, SSE4.2, AVX2+FMA |
| ARM64 (`aarch64-unknown-none`) | CI configured | `OPEN` | NEON via `fearless_simd` (untested on HW) |
| WebAssembly (`wasm32-unknown-unknown`) | CI configured | `OPEN` | SIMD128 via `fearless_simd` (untested) |
| Cortex-M (`thumbv7em-none-eabihf`) | CI configured | `OPEN` | Scalar fallback |
| RISC-V (`riscv32imac-unknown-none-elf`) | CI configured | `OPEN` | Explicit scalar fallback (no RVV claimed) |

See [`docs/PORTABILITY.md`](./docs/PORTABILITY.md).

## 10. Benchmarks

Measured on `AMD Ryzen 3 PRO 2300U` (4C/4T Zen 1, Windows 11 Pro, `rustc 1.100.0`, `--release`, 48 kHz / 32-sample block budget = `666.67 µs`):

- **FFT `RFFT + IFFT` (`SimdFixedFftBackend` vs `MicroFftBackend`)**: `N=64` **343.8 ns** (`2.08x`, SSE2), `N=128` **553.1 ns** (`2.85x`, AVX2), `N=256` **1,225.0 ns** (`2.51x`, AVX2).
- **`Hybrid(H128/T128)` @ 16,383 taps, 32-sample blocks**: `SimdFixedFft(SSE2)` with `set_tail_spreading(true)` reduces peak p99 callback time from `30.10 µs` (`4.52%`, MicroFFT burst) to **`5.20 µs` (`0.78%` of deadline)**.

See [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) for full tables and commands.

## 11. Examples

```bash
cargo run --release --example direct_fir
cargo run --release --example hybrid_convolution
cargo run --release --example backend_selection --all-features
cargo run --release --example multiple_instances
cargo run --release --example all_engines
cargo run --release --example no_std_static
```

## 12. Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md). Architecture, real-time, and evidence guides are in [`docs/`](./docs/).

## 13. Roadmap

See [`docs/ROADMAP.md`](./docs/ROADMAP.md) for planned post-`0.1.0` work (immutable shared IR spectra in Flash/ROM, non-uniform partitioning, pre-allocated crossfade, and hardware-in-the-loop verification).

## 14. License

Dual-licensed under [`MIT`](./LICENSE-MIT) or [`Apache-2.0`](./LICENSE-APACHE), at your option.
