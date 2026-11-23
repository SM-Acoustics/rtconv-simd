# ESP32-C6 Embedded Harness (`hardware/esp32c6`)

Independent `#![no_std]` bare-metal harness targeting **ESP32-C6 (`riscv32imac-unknown-none-elf`)**, excluded from the `crates.io` package (`Cargo.toml` `exclude = [..., "hardware/*"]`).

## Verification Status

- **`CROSS_COMPILE_VERIFIED`**: `YES` (`cargo check --target riscv32imac-unknown-none-elf`).
- **`DEVICE_VERIFIED`**: `NO` (physical ESP32-C6 execution and `mcycle` CSR measurements require a connected board and explicit owner authorization before flashing).
- **`REALTIME_MEASURED`**: `NO`.
- **`HARD_RT_CERTIFIED`**: `NO`.

## Reference Budget (`48 kHz`, `32`-sample block, `160 MHz` CPU clock)

- Block period: `32 / 48,000 = 666.67 µs`
- Gross cycle budget: `~106,667` cycles (shared with interrupts, DMA, and RTOS/HAL overhead).

## Cross-Compile Check

```bash
rustup target add riscv32imac-unknown-none-elf
cd hardware/esp32c6
cargo check --release --target riscv32imac-unknown-none-elf
```
