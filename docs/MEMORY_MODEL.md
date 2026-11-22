# Memory Model (`docs/MEMORY_MODEL.md`)

All core convolvers operate exclusively inside a single caller-owned `&mut [f32]` workspace slice sized via checked arithmetic (`memory_requirements`).

## Workspace Formulas (`f32` elements, 4 bytes each)

1. **`DirectConvolver`**: `2 * max_ir_len` (`ir[max_ir_len]` + `history[max_ir_len]`).
2. **`PartitionedConvolver`** (`B = block_size`, `N = 2B`, `bins = B + 1`, `P = ceil(max_ir_len / B)`):
   - IR spectra (Re + Im): `2 * P * bins`
   - FDL ring buffer (Re + Im): `2 * P * bins`
   - Previous input + time buffer + accumulator (Re + Im) + FFT scratch: `B + N + 2 * bins + fft.scratch_len_f32(N)`
3. **`HybridConvolver`** (`head_len = B_tail`):
   - `DirectConvolver(B_tail)` + `PartitionedConvolver(B_tail, max_ir_len - B_tail)` + `2 * B_tail` (staging in/out).

## Concrete Footprint Table (`SimdFixedFftBackend`, 48 kHz)

| Engine / Config | IR = 256 taps | IR = 2,048 taps | IR = 16,383 taps | External FFT Scratch | Peak Stack |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `DirectConvolver` | 2.00 KiB | 16.00 KiB | 127.99 KiB | 0 B | `< 128 B` |
| `Partitioned (B=32, FFT=64)` | 5.01 KiB | 33.88 KiB | 264.38 KiB | 256 B | `< 256 B` |
| `Partitioned (B=64, FFT=128)` | 5.01 KiB | 34.26 KiB | 261.76 KiB | 512 B | `< 256 B` |
| `Partitioned (B=128, FFT=256)` | 5.01 KiB | 35.76 KiB | 261.51 KiB | 1,024 B | `< 256 B` |
| `Hybrid (H128/T128, FFT=256)` | 4.76 KiB | 35.74 KiB | 261.49 KiB | 1,024 B | `< 256 B` |

- **Read-only (`const`) tables (`SimdFixedFftBackend`)**: `~3.2 KiB` total (`BITREV_32/64/128`, `TW_COS/SIN_4/8/16/32/64`, `R2C_COS/SIN_256`), residing in `.rodata` / Flash.
- **Stack note (`MicroFftBackend`)**: `MicroFftBackend` uses `0 B` external scratch but allocates `[Complex32; N]` on the stack inside `rfft_inverse` (`512 B` / `1,024 B` / `2,048 B` for `N = 64 / 128 / 256`). Embedded ISRs with tight stacks should prefer `SimdFixedFftBackend`.
