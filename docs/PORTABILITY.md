# Portability Matrix (`docs/PORTABILITY.md`)

Compilation, linking, and hardware execution are tracked separately per `SM-ENG-004` (*Proposed v1*). Conditional `#[cfg]` code is never reported as verified target support.

| Platform / Target Triple | Compile (`--no-default-features`) | Link / Test | Hardware Execution | SIMD Path |
| :--- | :--- | :--- | :--- | :--- |
| Windows x86-64 (`x86_64-pc-windows-gnu`) | `PROVEN` | `PROVEN` (`HOST_ONLY`) | `PROVEN` (`HOST_ONLY`, Ryzen 3 PRO 2300U) | SSE2, SSE4.2, AVX2+FMA |
| Linux x86-64 (`x86_64-unknown-linux-gnu`) | CI (`ci.yml`) | CI (`ci.yml`) | `OPEN` (CI runner only) | SSE2, SSE4.2, AVX2+FMA |
| ARM64 (`aarch64-unknown-none`) | CI (`embedded.yml`) | `OPEN` | `OPEN` | NEON via `fearless_simd` |
| WebAssembly (`wasm32-unknown-unknown`) | CI (`embedded.yml`) | `OPEN` | `OPEN` | SIMD128 (`-C target-feature=+simd128`) or fallback |
| Cortex-M4F/M7F (`thumbv7em-none-eabihf`) | CI (`embedded.yml`) | `OPEN` | `OPEN` | Scalar fallback |
| RISC-V (`riscv32imac-unknown-none-elf`) | CI (`embedded.yml`) | `OPEN` | `OPEN` | Explicit scalar fallback |

## ESP32-P4 / RISC-V Note

`fearless_simd 1.2.0` does **not** implement RISC-V Vector (RVV) or ESP32-P4 PIE SIMD intrinsics. On RISC-V targets, `rtconv-simd` explicitly selects the scalar fallback (`Capabilities::simd_active == false`, `fft_is_simd == false`). No RVV acceleration is claimed.
