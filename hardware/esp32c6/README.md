# ESP32-C6 Embedded Harness (`hardware/esp32c6`)

Independent `#![no_std]` bare-metal harness targeting **ESP32-C6 (`riscv32imac-unknown-none-elf`)**, excluded from the `crates.io` package (`Cargo.toml` `exclude = [..., "hardware/*"]`).

## Verification Status

- **`CROSS_COMPILE_VERIFIED`**: `YES` (`cargo build --release`, run in CI by `embedded.yml` job `esp32c6-harness` with `-D warnings`).
- **Workloads**: `FixedQ15DirectConvolver`, `DirectConvolver` (`f32` soft-float baseline), `FixedQ15PartitionedConvolver`, and `AutoConvolver` loading a compile-time raw `f32` `.bin` FIR artifact. On this FPU-less target `auto` selects the Q1.15 `FixedQ15NonUniformConvolver`; `HarnessReport::auto_ok` checks the impulse response. The same scenario runs on the host in `tests/auto_convolver.rs::esp32c6_harness_scenario_on_host`.
- **Stack**: the `AutoConvolver` value (~17 KiB, dominated by `i64` NUPC accumulators) and its workspace live on the `_start` stack in this compile-only harness. On a device, move them into `static` storage (for example `static_cell`) and confirm the stack high-water mark.
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
cargo build --release   # target set in .cargo/config.toml
```
