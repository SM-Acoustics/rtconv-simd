# Memory Model (`docs/MEMORY_MODEL.md`)

All core convolvers operate exclusively inside a single caller-owned workspace slice (`&mut [f32]` for floating-point engines, `&mut [i16]` for Q1.15/BFP engines) sized via checked arithmetic (`memory_requirements`).

## Workspace Formulas (`f32` elements, 4 bytes each)

1. **`DirectConvolver`**: `2 * max_ir_len` (`ir[max_ir_len]` + `history[max_ir_len]`).
2. **`PartitionedConvolver`** (`B = block_size ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}`, `N = 2B ∈ [64, 16384]`, `bins = B + 1`, `P = ceil(max_ir_len / B)`):
   - IR spectra (Re + Im): `2 * P * bins`
   - FDL ring buffer (Re + Im): `2 * P * bins`
   - Previous input + time buffer + accumulator (Re + Im) + FFT scratch: `B + N + 2 * bins + fft.scratch_len_f32(N)`
3. **`HybridConvolver`** (`head_len = B_tail ∈ {32..=8192}`):
   - `DirectConvolver(B_tail)` + `PartitionedConvolver(B_tail, max_ir_len - B_tail)` + `2 * B_tail` (staging in/out).
4. **`NonUniformConvolver`** (`feature = "nonuniform"`, up to `MAX_NONUNIFORM_STAGES = 8`, FDL `block_size ∈ {32..=8192}`):
   - Direct stage (`2 * L_0`) + single shared real input history ring (`max_s(max(2 * B_s, D_s + 1))` across FDL stages, replacing per-stage `2 * B_s` overlap buffers) + FDL spectral buffers (`4 * P * bins + 2 * bins`, reusing the newest FDL slot for in-place staged RFFT and `acc` for in-place staged IFFT) + power-of-two causal output delay rings (`2 * B_s` per FDL stage, branchless bitmask wrap) + shared FFT scratch (`2 * max_fft_size`). `NonUniformMemoryRequirements` reports exact `state_f32`, `spectral_f32`, `delay_f32`, `scratch_f32`, `total_f32`, and `total_bytes`.

## Fixed-Point Q1.15 / BFP Workspace (`i16` elements, 2 bytes each)

- **`FixedQ15DirectConvolver`** (`feature = "fixed-q15"`): `2 * max_ir_len` `i16` elements (**50% RAM** vs `DirectConvolver` `f32` — `128 B` for 32 taps, `2,048 B` for 512 taps).
- **`FixedQ15PartitionedConvolver`** (`feature = "fixed-fft"`, `B ∈ {32, 64, 128, 256, 512, 1024}`, `N = 2B ∈ [64, 2048]`):
  - Split `ir_re`/`ir_im`/`ir_exp`, FDL ring + per-slot BFP exponents, workspace-packed `i64` spectral accumulators (`8 * bins` `i16` words, which also host the in-place `4 * N + 4`-word inverse transform of the 32-bit-mantissa core), and the forward transform buffer (`2 * N + 4` `i16` words). Since issue #23 the temporary 16-bit spectrum (`2 * bins`) and time scratch (`N`) are gone and the FFT buffer shrank from `4 * N` to `2 * N + 4` words.
  - Plans/configs with `B > 1024` (`N > 2048`) are rejected explicitly with `ConvError::UnsupportedFftSize`.
- **`FixedQ15NonUniformConvolver`** (`feature = "fixed-nonuniform"`, FDL `B ∈ {32..=1024}`):
  - `FixedQ15NonUniformMemoryRequirements` breaks down `state_i16` (`2 * L_0` direct + single shared Q1.15 input history ring `max_s(max(2 * B_s, D_s + 1))`), `spectral_i16` (BFP spectra, exponents, and per-stage `i64` accumulators packed as `8 * bins = 4 * N + 8` `i16` words, which also host the in-place staged 32-bit-mantissa FFT (`2 * N + 4`) and IFFT (`4 * N + 4`, after normalizing the accumulators in place)), `delay_i16` (power-of-two `2 * B_s` Q1.23 output rings = `4 * B_s` `i16` words per FDL stage, summed before a single convergent RNE rounding), `scratch_i16` (`max_fft`, the shared overlap-save window — **zero separate FFT scratch and, since issue #23, no 16-bit spectrum scratch**), `total_i16`, `total_bytes`, and **`instance_bytes`**.
  - Because per-stage `i64` accumulators and staged BFP FFT/IFFT work arrays live in the caller `&mut [i16]` workspace, `instance_bytes` stays compact (**`< 4 KiB`**, down from `~16.5 KiB` in `0.2.0`) even with `N2048` stages, and is still counted by `NonUniformBudget` memory admission.

## ROM, External Scratch, and Peak Stack by FFT Backend

| Backend | Supported `N` | Twiddle / Bitrev ROM (`.rodata`) | External Scratch | Peak Stack |
| :--- | :--- | ---: | :--- | :--- |
| **`SimdFixedFftBackend`** (`f32`) | `64..=16384` | **`16,388 B`** (`[f32; 4097]` quarter-wave, `0 B` bitrev ROM) | `N` `f32` (`256 B`–`64.00 KiB`); `0 B` extra for staged | `< 256 B` |
| **`MicroFftBackend`** (`f32`) | `64..=16384` | **`16,380 B`** (`microfft 0.6.0` `size-16384` sine table) | `N` `f32` (`256 B`–`64.00 KiB`) | `< 256 B` |
| **`PhastFftBackend`** (`f32`, `std`) | `64..=16384` | Precomputed `PlannerR2c32` (allocated once in `new()`) | `N` `f32` (`256 B`–`64.00 KiB`) | `< 256 B` |
| **`fft_q15` (`*_with_scratch`, public stand-alone API)** | `64..=2048` | **`1,026 B`** (`[i16; 513]` quarter-wave, `0 B` bitrev ROM) | `2 * N` `i32` (`512 B`–`16.00 KiB`) | `< 256 B` |
| **`fft_q31` core (`FixedQ15*` engines, issue #23)** | `64..=2048` | **`2,052 B`** (`[i32; 513]` Q1.31 quarter-wave) | forward `N + 2`, inverse `2 * N + 2` `i32` words (in the engine workspace) | `336 B` partitioned / `496 B` NUPC callback chain (static, RV32IMAC) |
| **`fft_q15` (legacy `rfft_q15_bfp`)** | `64..=256` only | shares the `1,026 B` ROM (`N > 256` → `UnsupportedFftSize`) | `0 B` | `~2 KiB` (`[i32; 256] × 2`) |

- **`SimdFixedFftBackend` twiddles**: generated in `f64` and rounded to `f32` under IEEE 754 round-to-nearest-even in `build.rs`, stored as a single `static` quarter-wave cosine table (`TW_COS_Q: [f32; 4097]`) strided by power of two. Bit-reversal indices are computed with `usize::reverse_bits()`.
- **`MicroFftBackend` zero-stack inverse**: uses the Hermitian real-FFT identity (`rfft` on time-reversed real-symmetric input inside the caller's `N`-float scratch), eliminating the `[f32; N]` and `[Complex32; N]` stack arrays that previously overflowed MCU stacks for large `N`.
- **`FixedQ15PartitionedConvolver` & `FixedQ15NonUniformConvolver` peak stack**: **`< 512 B`** across all sizes `N64..=N2048` (both `i64` spectral accumulators and FFT `re`/`im` work arrays live in the caller `&mut [i16]` workspace).

## Concrete Footprint Table (`SimdFixedFftBackend` / `MicroFftBackend` / `fft_q15`)

| Engine / Config | IR = 2,048 taps | IR = 16,384 taps | IR = 65,536 taps | IR = 131,072 taps | External FFT Scratch | Peak Stack |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DirectConvolver` (`f32`) | 16.00 KiB | 128.00 KiB | 512.00 KiB | 1,024.00 KiB | 0 B | `< 128 B` |
| `FixedQ15DirectConvolver` (`i16`) | 8.00 KiB | 64.00 KiB | 256.00 KiB | 512.00 KiB | 0 B | `< 128 B` |
| `Partitioned (B=32, N=64)` | 33.88 KiB | 264.88 KiB | 1,056.88 KiB | 2,112.88 KiB | 256 B | `< 256 B` |
| `Partitioned (B=128, N=256)` | 35.76 KiB | 261.51 KiB | 1,035.51 KiB | 2,067.51 KiB | 1,024 B | `< 256 B` |
| `Partitioned (B=512, N=1024)` | 46.07 KiB | 270.51 KiB | 1,040.01 KiB | 2,066.01 KiB | 4,096 B | `< 256 B` |
| `Partitioned (B=2048, N=4096)` | 88.02 KiB | 312.13 KiB | 1,080.51 KiB | 2,105.01 KiB | 16,384 B | `< 256 B` |
| `Partitioned (B=8192, N=16384)` | 352.02 KiB | 480.04 KiB | 1,248.13 KiB | 2,272.26 KiB | 65,536 B | `< 256 B` |
| `Hybrid (H128/T128, N=256)` | 35.74 KiB | 261.49 KiB | 1,035.49 KiB | 2,067.49 KiB | 1,024 B | `< 256 B` |
| `Hybrid (H512/T512, N=1024)` | 46.05 KiB | 270.49 KiB | 1,039.99 KiB | 2,065.99 KiB | 4,096 B | `< 256 B` |
| `Hybrid (H2048/T2048, N=4096)` | 32.00 KiB | 312.12 KiB | 1,080.49 KiB | 2,104.99 KiB | 16,384 B | `< 256 B` |
| `NonUniform (D32/32/64/128)` | 41.70 KiB | 267.45 KiB | 1,041.45 KiB | 2,073.45 KiB | 1,024 B | `< 256 B` |
| `NonUniform (D32/32/128/512/2k/8k)` | — | 618.49 KiB | 1,386.59 KiB | 2,410.71 KiB | 65,536 B | `< 256 B` |
| `FixedQ15Partitioned (B=128, N=256)` | 21.71 KiB | 135.02 KiB | — | — | 2,048 B (`in ws`) | `< 512 B` |
| `FixedQ15Partitioned (B=512, N=1024)` | 38.07 KiB | 150.39 KiB | — | — | 8,192 B (`in ws`) | `< 512 B` |
| `FixedQ15Partitioned (B=1024, N=2048)` | 60.04 KiB | 172.21 KiB | — | — | 16,384 B (`in ws`) | `< 512 B` |
| `FixedQ15NonUniform (D32/32/64/128)` | 24.49 KiB | 137.81 KiB | — | — | 0 B extra (`in acc`) | `< 512 B` |
| `FixedQ15NonUniform (preset_esp32_c6_q15)` | 26.73 KiB | 169.37 KiB | — | — | 0 B extra (`in acc`) | `< 512 B` |

*Note on PR-C NUPC ring & scratch savings (Issue #16)*: Clearance-aware deferral (`defer_s = slack_s.saturating_sub(B_s - 1)`) shrinks every FDL output delay ring from `D_s + B_s` to power-of-two `2 * B_s` (`4 * B_s` `i16` words in Q1.23) while replacing all per-stage input overlap buffers with a single shared input ring of length `max_s(max(2 * B_s, D_s + 1))`. Compared to PR-A/PR-B, this saves `21.25 KiB` on `NonUniform (D32..8k)`, `37.25 KiB` on `NonUniform (D32..4k)`, `−56.9%` delay-ring RAM on high-offset `B = 8192 @ 30000` plans, and `−75.2%` delay-ring RAM plus **`−16.00 KiB`** shared FFT scratch on `FixedQ15NonUniform` (`N2048`) because staged BFP FFT/IFFT runs in-place inside each stage's `8 * (B_s + 1)` accumulator.

## Partition Optimizer Scratch & Pre-Computed MCU Presets (`src/nonuniform/optimizer.rs`, Class B)

The Viterbi partition optimizer runs strictly at setup time (Class B, never inside the audio callback) with **zero heap allocation** and `< 512 B` peak stack, storing all DP tables and packed `u32` back-pointers inside a caller-supplied `&mut [OptimizerScratchCell]` (`8 B` per cell) sized by `const fn optimizer_scratch_len(ir_len, max_block_size)`:

| Target IR Length | `max_block_size = 8192` (`f32`) | `max_block_size = 1024` (`Q1.15`) | `preset_esp32_p4_f32` / `preset_esp32_c6_q15` |
| ---: | ---: | ---: | ---: |
| 1,024 taps | `89.3 KiB` | `14.1 KiB` | **`0 B`** |
| 4,096 taps | `123.3 KiB` | `35.1 KiB` | **`0 B`** |
| 16,384 taps | `246.3 KiB` | `119.1 KiB` | **`0 B`** |
| 65,536 taps | `738.3 KiB` | `455.1 KiB` | **`0 B`** |

- Inside `AutoConvolver::prepare_f32` / `prepare_q15` / `prepare_q31_with`, the Viterbi DP runs **in-place inside the engine's own workspace** (`optimize_in_f32_scratch` / `optimize_in_i16_scratch`, capped at `B <= max(32, L/4)` and `4` / `3` FDL stages) before the convolver zeroes and initializes the workspace, requiring **zero extra permanent RAM** beyond the convolver workspace itself.
- Memory-constrained MCUs can bypass the DP entirely with `preset_esp32_p4_f32(ir_len, host_block_max)` or `preset_esp32_c6_q15(ir_len, host_block_max)` (`0 B` scratch).

## Automatic Layer (`auto`, default feature)

- `auto::select_f32` evaluates `Direct`, `Hybrid(B)` (`B ∈ {32..=8192}`), and Strategy-2 `NonUniform` under `CostModel::for_target_f32()`.
- `auto::select_q15` selects `FixedDirect` (`<= 128` taps) or cost-optimized `FixedNonUniform` (`B <= 1024 / N2048`, `CostModel::esp32_c6_q15_estimate()`) above `128` taps.
- `auto::workspace_len(max_taps)` (`const fn`) returns the elements of `auto::NativeElem` (`f32` with an FPU, `i16` without) needed for any IR of up to `max_taps` taps plus IR decode scratch (`max_taps` `f32`, or `2 * max_taps` `i16` for Q1.31 taps).
- `worst_engine_len_f32` evaluates in `O(1)` so `AutoWorkspace::<{ workspace_len(131_072) }>` compiles cleanly in `const` generics without hitting Rust's `const`-eval step limit (`tests/auto_convolver.rs::const_generic_workspace_len_evaluates_at_131072_taps`).
- `auto::engine_len_f32` / `engine_len_q15` are **exact** for `Direct`, `Hybrid`, and `FixedDirect`, and return a **mathematically proven tight upper bound** for cost-optimized `NonUniform` / `FixedNonUniform` (covering both in-place Viterbi DP scratch and the resulting convolver workspace, verified in `tests/auto_convolver.rs::const_sizing_matches_engine_requirements_exactly`).
- `stereo_workspace_len` doubles `workspace_len`. The IR scratch tail of the workspace is used only during preparation.
