# WO-RTCONV-004F — ESP32-C6 Hardware Validation Harness

**Work Order**: `RTCONV-004F`  
**Branch**: `test/rtconv-004f-esp32c6`  
**Target**: `riscv32imac-unknown-none-elf` (ESP32-C6, FPU-less 32-bit RISC-V @ 160 MHz)  
**Harness Location**: `hardware/esp32c6/` (independent Cargo package, excluded from `crates.io` via `Cargo.toml` `exclude = [..., "hardware/*"]`, zero dependency from core crate).

---

## 1. Gate F Classification (Explicit Separation)

| Level | Status | Evidence |
| :--- | :--- | :--- |
| **`CROSS_COMPILE_VERIFIED`** | **`PASS`** | `cargo check --release --target riscv32imac-unknown-none-elf` in `hardware/esp32c6/` and `cargo check --lib --no-default-features --features fixed-nonuniform --target riscv32imac-unknown-none-elf` in root (`rustc 1.99.1`, WSL Ubuntu 24.04) -> `EXIT 0`. |
| **`DEVICE_VERIFIED`** | **`NO` (`NOT_VERIFIED`)** | No physical ESP32-C6 board flashed or executed; flashing is prohibited without explicit owner authorization. No physical cycle counts are simulated or fabricated. |
| **`REALTIME_MEASURED`** | **`NO` (`NOT_VERIFIED`)** | Awaits authorized physical ESP32-C6 `mcycle` CSR run. |
| **`HARD_RT_CERTIFIED`** | **`NO`** | Not claimed. |

## 2. Harness Scope & Reference Budget

- `hardware/esp32c6/src/main.rs` (`#![no_std]`, `#![no_main]`, zero heap, zero I2S) validates `FixedQ15DirectConvolver` (64 taps), software `DirectConvolver` (`f32` baseline, 64 taps), and `FixedQ15PartitionedConvolver` (`B = 32 / N64`, 64 taps) against exact synthetic impulse expectations on `32`-sample blocks.
- Nominal `48 kHz` / `32`-sample block period: `666.67 µs` (`~106,667` gross CPU cycles at `160 MHz`, shared with interrupts/DMA/system overhead).
- Physical hardware logs will be saved under `hardware/esp32c6/measurements/` once authorized and executed on a connected ESP32-C6.
