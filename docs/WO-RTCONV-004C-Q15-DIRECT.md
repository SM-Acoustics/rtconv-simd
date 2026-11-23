# WO-RTCONV-004C — Fixed-Point Q1.15 Direct FIR (`FixedQ15DirectConvolver`)

**Work Order**: `RTCONV-004C`  
**Branch**: `feat/rtconv-004c-fixed-q15`  
**Feature Gate**: `fixed-q15` (`default = []` preserved; `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap, zero C/FFI, zero `esp-hal` dependency)  
**Verification Scope**: `HOST_TESTED` (`PROVEN` on Windows x86-64 `rustc 1.100.0` + MSRV `1.89.0`) & `CROSS_COMPILE_VERIFIED` (`riscv32imac-unknown-none-elf`, WSL Ubuntu 24.04 `rustc 1.99.1`); `DEVICE_VERIFIED = NO` (physical ESP32-C6 execution tracked in `RTCONV-004F`).

---

## 1. Numeric Contract (Q1.15)

- **Format**: Signed `i16` in `[-32768, +32767]` representing `[-1.0, 1.0 - 2^-15]` (1 LSB = `2^-15 ≈ 3.0518e-5`, dynamic range `~90.3 dB`).
- **Products & Accumulation**: `16 × 16 -> Q2.30`, accumulated in `i64` (`acc += (x as i64) * (h as i64)`), guaranteeing zero intermediate accumulator overflow for up to `2^33` taps.
- **Rounding & Saturation**: Explicit round-half-up `(acc + 16384) >> 15` followed by symmetric saturation to `[i16::MIN, i16::MAX]`, with runtime saturation telemetry (`saturation_count()`).
- **RV32IMAC Cost Note**: On 32-bit RISC-V (`RV32IMAC`), 64-bit accumulation is not free — each tap executes `mul` + `mulh` (or 32-bit product sign-extension) plus a 64-bit two-register add (`add` + `sltu` + `add`), avoiding any software `f32` emulation while keeping exact headroom.

---

## 2. Numerical & Zero-Allocation Verification

- **`tests/fixed_q15_reference.rs`** (independent `f64` oracle):
  - Extreme coefficients (`i16::MIN * i16::MIN -> i16::MAX` with `saturation_count() == 1`), unity/attenuation, rounding boundary tests (`±16384`, `16383`, `-16385`), deliberate positive/negative overflow saturation, and 512-tap decaying random IR:
  - Measured 512-tap error vs `f64`: `max_abs <= 1 LSB (3.05e-5)`, `RMS = 8.911e-6` (at theoretical floor `2^-15 / sqrt(12) ≈ 8.81e-6`), `SNR = 69.55 dB`, bit-identical `reset()`.
- **`tests/no_alloc_rt.rs`**: Standalone `harness = false` verifier proves `0` allocations / `0` deallocations in `fixed_q15_direct::{prepare, process_sample, process_block, reset, metadata}` on both `1.100.0` and `1.89.0`.
- **Cross-Compilation (`CROSS_COMPILE_VERIFIED`)**: `cargo check --lib --no-default-features --features fixed-q15 --target riscv32imac-unknown-none-elf` -> `EXIT 0` (`rustc 1.99.1`, WSL Ubuntu 24.04).

---

## 3. Host Benchmark (`HOST_ONLY` — x86-64 with Hardware FPU)

**Host**: AMD Ryzen 3 PRO 2300U, Windows 11 Pro `10.0.26300`, `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `x86_64-pc-windows-gnu`, `48 kHz`, `block = 32` (`666.67 µs` budget), `cargo bench --bench fixed_q15_bench --features fixed-q15`.

| Taps | Q15 Direct (µs / % budget) | Q15 RAM | `f32` Scalar (µs) | `f32` SIMD (µs) | `f32` RAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.35 µs (0.20%) | 128 B | 0.61 µs | 0.65 µs | 256 B |
| 64 | 2.53 µs (0.38%) | 256 B | 1.17 µs | 0.77 µs | 512 B |
| 128 | 5.83 µs (0.87%) | 512 B | 3.77 µs | 1.14 µs | 1,024 B |
| 256 | 9.80 µs (1.47%) | 1,024 B | 8.43 µs | 2.15 µs | 2,048 B |
| 512 | 19.34 µs (2.90%) | 2,048 B | 17.45 µs | 4.21 µs | 4,096 B |

*Note*: On x86-64 with hardware FPU/SSE2, `f32` SIMD is faster while Q15 halves RAM footprint (`50%`). The primary compute advantage of Q15 over software `f32` applies to FPU-less RV32IMAC targets (evaluated in `RTCONV-004F`).
