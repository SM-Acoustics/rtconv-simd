# Portability Matrix (`docs/PORTABILITY.md`)

Compilation, MSRV, cross-compilation, host testing, and hardware execution are tracked separately per `SM-ENG-004` (*Proposed v1*). Conditional `#[cfg]` code is never reported as verified target support.

| Platform / Target Triple | Compile (`--no-default-features`) | Link / Test | Hardware Execution | SIMD Path |
| :--- | :--- | :--- | :--- | :--- |
| Windows x86-64 GNU (`x86_64-pc-windows-gnu`) | `PROVEN` (`MSRV_VERIFIED` 1.89.0 + 1.100.0) | `HOST_TESTED` | `HOST_TESTED` (Ryzen 3 PRO 2300U) | SSE2, SSE4.2, AVX2+FMA |
| Windows x86-64 MSVC (`x86_64-pc-windows-msvc`) | CI (`ci.yml`, `windows-latest`) | CI (`ci.yml`) | `PENDING_REMOTE_CI` | SSE2, SSE4.2, AVX2+FMA |
| Linux x86-64 (`x86_64-unknown-linux-gnu`) | CI (`ci.yml`, `ubuntu-latest`) | CI (`ci.yml`) | `PENDING_REMOTE_CI` | SSE2, SSE4.2, AVX2+FMA |
| macOS Apple Silicon (`aarch64-apple-darwin`) | CI (`ci.yml`, `macos-latest`) | CI (`ci.yml`) | `PENDING_REMOTE_CI` | NEON via `fearless_simd` |
| ARM64 bare-metal (`aarch64-unknown-none`) | CI (`embedded.yml`) | `OPEN` | `OPEN` (`DEVICE_VERIFIED: NO`) | NEON via `fearless_simd` |
| WebAssembly (`wasm32-unknown-unknown`) | CI (`embedded.yml`) | `OPEN` | `OPEN` (`DEVICE_VERIFIED: NO`) | SIMD128 (`-C target-feature=+simd128`) or fallback |
| Cortex-M4F/M7F (`thumbv7em-none-eabihf`) | CI (`embedded.yml`) | `OPEN` | `OPEN` (`DEVICE_VERIFIED: NO`) | Scalar fallback |
| RISC-V (`riscv32imac-unknown-none-elf`) | CI (`embedded.yml`) | `OPEN` | `OPEN` (`DEVICE_VERIFIED: NO`) | Explicit scalar fallback |

## ESP32-P4 / RISC-V Note

`fearless_simd 1.2.0` does **not** implement RISC-V Vector (RVV) or ESP32-P4 PIE SIMD intrinsics. On RISC-V targets, `rtconv-simd` explicitly selects the scalar fallback (`Capabilities::simd_active == false`, `fft_is_simd == false`). No RVV acceleration is claimed.
