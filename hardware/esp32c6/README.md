# ESP32-C6 Embedded Harness (`hardware/esp32c6`)

Independent `#![no_std]` bare-metal harness targeting **ESP32-C6 (`riscv32imac-unknown-none-elf`)**, excluded from the `crates.io` package (`Cargo.toml` `exclude = [..., "hardware/*"]`).

## Verification Status

- **`CROSS_COMPILE_VERIFIED`**: `YES` (`cargo build --release`, run in CI by `embedded.yml` job `esp32c6-harness` with `-D warnings`).
- **Workloads**: `FixedQ15DirectConvolver`, `FixedQ15PartitionedConvolver`, and `AutoConvolver` loading a compile-time raw `f32` `.bin` FIR artifact. On this FPU-less target `auto` selects the Q1.15 `FixedQ15NonUniformConvolver`; `HarnessReport::auto_ok` checks the impulse response. The same scenario runs on the host in `tests/auto_convolver.rs::esp32c6_harness_scenario_on_host`. The `f32` `DirectConvolver` soft-float comparison is opt-in: `cargo build --release --features soft-float-baseline`.
- **No soft-float, no dead `f32` path** (issue #32): the default build links none of `TW_COS_Q`, `SimdFixedFftBackend`, `POWER_OF_FIVE_128`, `__mulsf3`, `__muldf3` or any other soft-float routine, and `.text + .rodata` stays within the 0.2.0 size (81,196 B). CI enforces both with `scripts/check-no-fpu-binary.sh` and writes `llvm-size -A` to the job summary. Measured: `.text` 43,362 B + `.rodata` 6,656 B = 50,018 B (`rustc 1.97.0`), down from 119,604 B.
- **Stack**: the `AutoConvolver` value (1,728 B on `riscv32imac`; the NUPC `i64` accumulators live in the workspace since #14) and its workspace live on the `_start` stack in this compile-only harness. On a device, move the workspace into `static` storage (for example `static_cell`) and confirm the stack high-water mark. Static call-graph bound of the DSP callbacks (sum of `addi sp,sp,-N` frames along the deepest direct-call path, compile-only, not a WCET/HWM measurement): `FixedQ15PartitionedConvolver::process_block` 320 B, `FixedQ15NonUniformConvolver::process_block` 496 B, through `AutoConvolver::process_i16` 544 B (pre-#23: 448 / 624 / 672 B; panic paths excluded).
- **`DEVICE_VERIFIED`**: `NO` (physical ESP32-C6 execution and `mcycle` CSR measurements require a connected board and explicit owner authorization before flashing).
- **`REALTIME_MEASURED`**: `NO`.
- **`HARD_RT_CERTIFIED`**: `NO`.

## Reference Budget (`48 kHz`, `32`-sample block, `160 MHz` CPU clock)

- Block period: `32 / 48,000 = 666.67 µs`
- Gross cycle budget: `~106,667` cycles (shared with interrupts, DMA, and RTOS/HAL overhead).

## Cross-Compile Check

```bash
rustup target add riscv32imac-unknown-none-elf
rustup component add llvm-tools
cd hardware/esp32c6
cargo build --release   # target set in .cargo/config.toml
bin="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin"
LLVM_NM="$bin/llvm-nm" LLVM_SIZE="$bin/llvm-size" \
  ../../scripts/check-no-fpu-binary.sh target/riscv32imac-unknown-none-elf/release/rtconv-esp32c6-harness
```
