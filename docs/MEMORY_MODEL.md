# Memory Model (`docs/MEMORY_MODEL.md`)

All core convolvers operate exclusively inside a single caller-owned workspace slice (`&mut [f32]` for floating-point engines, `&mut [i16]` for Q1.15/BFP engines) sized via checked arithmetic (`memory_requirements`).

## Workspace Formulas (`f32` elements, 4 bytes each)

1. **`DirectConvolver`**: `2 * max_ir_len` (`ir[max_ir_len]` + `history[max_ir_len]`).
2. **`PartitionedConvolver`** (`B = block_size`, `N = 2B`, `bins = B + 1`, `P = ceil(max_ir_len / B)`):
   - IR spectra (Re + Im): `2 * P * bins`
   - FDL ring buffer (Re + Im): `2 * P * bins`
   - Previous input + time buffer + accumulator (Re + Im) + FFT scratch: `B + N + 2 * bins + fft.scratch_len_f32(N)`
3. **`HybridConvolver`** (`head_len = B_tail`):
   - `DirectConvolver(B_tail)` + `PartitionedConvolver(B_tail, max_ir_len - B_tail)` + `2 * B_tail` (staging in/out).
4. **`NonUniformConvolver`** (`feature = "nonuniform"`, up to `MAX_NONUNIFORM_STAGES = 8`):
   - Per-stage Direct (`2 * L_0`) + FDL stage workspaces (`ir_spectra + fdl + staging + causal delay ring of ir_offset + B_s`) + shared FFT scratch (`NonUniformMemoryRequirements` reports exact `total_f32` and `total_bytes`).

## Fixed-Point Q1.15 / BFP Workspace (`i16` elements, 2 bytes each)

- **`FixedQ15DirectConvolver`** (`feature = "fixed-q15"`): `2 * max_ir_len` `i16` elements (**50% RAM** vs `DirectConvolver` `f32` — `128 B` for 32 taps, `2,048 B` for 512 taps).
- **`FixedQ15PartitionedConvolver`** (`feature = "fixed-fft"`): Split `ir_re`/`ir_im`/`ir_exp`, FDL ring + per-slot BFP exponents, and caller `&mut [i16]` scratch; shares a **`512 B`** `const` twiddle ROM (`TWIDDLE_COS_256` + `TWIDDLE_SIN_256`) across `N64`/`N128`/`N256`.
- **`FixedQ15NonUniformConvolver`** (`feature = "fixed-nonuniform"`): `FixedQ15NonUniformMemoryRequirements` breaks down `state_i16`, `spectral_i16`, `delay_i16` (each causal-ring sample is a Q1.23 `i32` stored as **2** `i16` words, so stages are summed before a single rounding), `scratch_i16`, `total_i16`, `total_bytes`, and **`instance_bytes`**. The convolver value itself embeds `2 × 8 × 129` `i64` spectral accumulators (about **16.5 KiB**); this memory is not part of the workspace, is counted by `NonUniformBudget` memory admission, and should live in a `static` on MCUs.

## Concrete Footprint Table (`SimdFixedFftBackend`, 48 kHz)

| Engine / Config | IR = 256 taps | IR = 2,048 taps | IR = 16,383 taps | External FFT Scratch | Peak Stack |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `DirectConvolver` (`f32`) | 2.00 KiB | 16.00 KiB | 127.99 KiB | 0 B | `< 128 B` |
| `FixedQ15DirectConvolver` (`i16`) | 1.00 KiB | 8.00 KiB | 64.00 KiB | 0 B | `< 128 B` |
| `Partitioned (B=32, FFT=64)` | 5.01 KiB | 33.88 KiB | 264.38 KiB | 256 B | `< 256 B` |
| `Partitioned (B=64, FFT=128)` | 5.01 KiB | 34.26 KiB | 261.76 KiB | 512 B | `< 256 B` |
| `Partitioned (B=128, FFT=256)` | 5.01 KiB | 35.76 KiB | 261.51 KiB | 1,024 B | `< 256 B` |
| `Hybrid (H128/T128, FFT=256)` | 4.76 KiB | 35.74 KiB | 261.49 KiB | 1,024 B | `< 256 B` |
| `NonUniform (D32/32/64/128)` | — | 43.70 KiB | 269.45 KiB | 1,024 B | `< 256 B` |

- **Read-only (`const`) tables (`SimdFixedFftBackend`)**: `~3.2 KiB` total (`BITREV_32/64/128`, `TW_COS/SIN_4/8/16/32/64`, `R2C_COS/SIN_256`), residing in `.rodata` / Flash.
- **Read-only (`const`) tables (`fft_q15`)**: `512 B` total (`TWIDDLE_COS_256`, `TWIDDLE_SIN_256`).
- **Stack note (`fft_q15`)**: `rfft_q15_bfp` / `irfft_q15_bfp` use two `[i32; 256]` work arrays (**2 KiB**) per call. `FixedQ15PartitionedConvolver::process_block` adds a 2 KiB `[i64; 129] × 2` accumulator, and `FixedQ15NonUniformConvolver` adds a 1 KiB `[i32; 256]` wide IFFT output, for a peak of about **4–5 KiB** of stack in the fixed-point engines.
- **Stack note (`MicroFftBackend`)**: `MicroFftBackend` uses `0 B` external scratch but allocates `[Complex32; N]` on the stack inside `rfft_inverse` (`512 B` / `1,024 B` / `2,048 B` for `N = 64 / 128 / 256`). Embedded ISRs with tight stacks should prefer `SimdFixedFftBackend` or `fft_q15`.

## Automatic Layer (`auto`, default feature)

- `auto::workspace_len(max_taps)` (`const fn`) returns the elements of `auto::NativeElem` (`f32` with an FPU,
  `i16` without) needed for any IR of up to `max_taps` taps. It is the worst case over the engines the cost
  model can select for shorter IRs, plus the IR decode scratch (`max_taps` `f32`, or `2 × max_taps` `i16` for
  Q1.31 taps).
- `auto::engine_len_f32` / `engine_len_q15` mirror the engines' `memory_requirements` exactly. This is
  asserted for every length 1–4500 plus 8191, 8192, 16383 and 65536 in `tests/auto_convolver.rs`.
- `AutoWorkspace<{ workspace_len(N) }>` is a `const`-constructible array for `static` placement.
  `stereo_workspace_len` doubles it.
- The IR scratch tail of the workspace is used only during preparation.
