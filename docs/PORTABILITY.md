# Portability Matrix (`docs/PORTABILITY.md`)

Compilation, MSRV, cross-compilation, host testing, and hardware execution are tracked separately per `SM-ENG-004` (*Proposed v1*). Conditional `#[cfg]` code is never reported as verified target support.

| Platform / Target Triple | Compile (`--no-default-features` & `--all-features`) | Link / Test | Hardware Execution | SIMD / Numeric Path |
| :--- | :--- | :--- | :--- | :--- |
| Windows x86-64 GNU (`x86_64-pc-windows-gnu`) | `PROVEN` (`MSRV_VERIFIED` `1.89.0` + `1.100.0`) | `HOST_TESTED` | `HOST_TESTED` (Ryzen 3 PRO 2300U) | SSE2, SSE4.2, AVX2+FMA + Q1.15/BFP |
| Windows x86-64 MSVC (`x86_64-pc-windows-msvc`) | `CI_VERIFIED` (`ci.yml`, `windows-latest`) | `CI_VERIFIED` (incl. `no_alloc_rt`) | `CI_HOST_ONLY` | SSE2, SSE4.2, AVX2+FMA + Q1.15/BFP |
| Linux x86-64 (`x86_64-unknown-linux-gnu`) | `CI_VERIFIED` (`stable` + MSRV `1.89.0`, `ubuntu-latest`) | `CI_VERIFIED` (incl. `no_alloc_rt`) | `CI_HOST_ONLY` | SSE2, SSE4.2, AVX2+FMA + Q1.15/BFP |
| macOS Apple Silicon (`aarch64-apple-darwin`) | `CI_VERIFIED` (`ci.yml`, `macos-latest`) | `CI_VERIFIED` (incl. standalone `no_alloc_rt`) | `CI_HOST_ONLY` | NEON via `fearless_simd` + Q1.15/BFP |
| ARM64 bare-metal (`aarch64-unknown-none`) | `CROSS_COMPILE_VERIFIED` (`embedded.yml`) | `OPEN` | `OPEN` (`DEVICE_VERIFIED: NO`) | NEON via `fearless_simd` |
| WebAssembly (`wasm32-unknown-unknown`) | `CROSS_COMPILE_VERIFIED` (`embedded.yml`) | `OPEN` | `OPEN` (`DEVICE_VERIFIED: NO`) | SIMD128 (`-C target-feature=+simd128`) or fallback |
| Cortex-M4F/M7F (`thumbv7em-none-eabihf`) | `CROSS_COMPILE_VERIFIED` (`embedded.yml`) | `OPEN` | `OPEN` (`DEVICE_VERIFIED: NO`) | Scalar fallback / Q1.15/BFP |
| RISC-V / ESP32-C6 (`riscv32imac-unknown-none-elf`) | `CROSS_COMPILE_VERIFIED` (`embedded.yml`: core, default `auto`, `fixed-nonuniform`, `-D warnings`; `hardware/esp32c6/` harness) | host mirror (`tests/auto_convolver.rs::esp32c6_harness_scenario_on_host`) | `OPEN` (`DEVICE_VERIFIED: NO`, harness ready) | `auto` → Q1.15/BFP; explicit scalar fallback for `f32` engines |
| RISC-V / ESP32-P4 (`riscv32imafc-unknown-none-elf`) | `CROSS_COMPILE_VERIFIED` (`embedded.yml`: core, default `auto`, `fixed-nonuniform`, `-D warnings`) | `OPEN` | `OPEN` (`DEVICE_VERIFIED: NO`) | `auto` → `f32` (hardware single-precision FPU, scalar; PIE SIMD unused) |
| Cortex-M0/M0+ (`thumbv6m-none-eabi`) | `CROSS_COMPILE_VERIFIED` (`embedded.yml`) | `OPEN` | `OPEN` | `auto` → Q1.15/BFP |

## Automatic Numeric Domain (`auto::NativeElem`)

`build.rs` emits `cfg(rtconv_no_fpu)` when `f32` would be software-emulated. On stable Rust, `cfg(target_feature = "f")` is **not** exposed for RISC-V, so `riscv32imac` and `riscv32imafc` have identical cfgs. The ISA letters of the target triple are therefore parsed (`f`, `d` or `g` ⇒ FPU). For ARM the rule is the `eabihf` ABI or VFP features; ESP32-S2, AVR and MSP430 are treated as FPU-less. Classification was probed with `cargo build -vv` for `riscv32imac` (Q1.15), `riscv32imafc` (`f32`), `thumbv6m` (Q1.15), `thumbv7em-none-eabihf` (`f32`) and `x86_64` (`f32`). The `native-q15` feature forces Q1.15.

## RISC-V / ESP32-C6 / ESP32-P4 Note

- `fearless_simd 1.2.0` does **not** implement RISC-V Vector (RVV) or ESP32-P4 PIE SIMD intrinsics. On RISC-V targets, the `f32` path explicitly selects the scalar fallback (`Capabilities::simd_active == false`, `fft_is_simd == false`). No RVV acceleration is claimed.
- On FPU-less `RV32IMAC` targets (such as ESP32-C6), the `fixed-q15`, `fixed-fft`, and `fixed-nonuniform` features provide an FPU-free Q1.15 / Block Floating Point stack verified to cross-compile cleanly (`CROSS_COMPILE_VERIFIED`). Physical `mcycle` profiling awaits authorized device execution under `hardware/esp32c6/`.
