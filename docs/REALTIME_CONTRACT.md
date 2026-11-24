# Real-Time Contract (`docs/REALTIME_CONTRACT.md`)

Aligned with `SM-ENG-002` (*Proposed v1*): `prepare()` belongs to **Class B** (off-callback bounded preparation); `process_sample()`, `process_block()`, and `reset()` belong to **Class A** (hard-deadline audio processing).

## Class A Invariants (`PROVEN`, `HOST_ONLY`)

1. **Zero heap activity**: Verified via serialized `GlobalAlloc` (`tests/no_alloc_rt.rs`) for `DirectConvolver`, `PartitionedConvolver` (`N64..=N16384`), `HybridConvolver` (`BURST` and `SPREAD`), `NonUniformConvolver` (up to `N16384`, including staged `rfft_forward_step` / `rfft_inverse_step`), `NonUniformPlan::optimize`, `optimize_block_mode`, `preset_esp32_*`, `FixedQ15DirectConvolver`, `FixedQ15PartitionedConvolver` (`N64..=N2048`), `FixedQ15NonUniformConvolver` (up to `N2048`, including staged `rfft_q15_bfp_step_with_scratch` / `irfft_q15_bfp_step_with_scratch`), `AutoConvolver` (including `AutoEngine::NonUniform` in-place DP), `MicroFftBackend`, `SimdFixedFftBackend`, and `PhastFftBackend` (with pre-initialized planners up to `N16384`).
2. **Zero blocking operations**: No mutexes, channels, threads, syscalls, file I/O, or logging in `src/`.
3. **Zero runtime CPUID, partition optimization, or FFT planning in callback**: `SimdBackend`, `NonUniformPlan::optimize` (Class B), and `PhastFftBackend` planners are resolved once prior to streaming.
4. **Bounded per-callback work**:
   - `DirectConvolver`: `O(B · L)` multiply-adds per block of `B` samples and IR length `L`.
   - `PartitionedConvolver`: `1 RFFT(2B) + P · (B + 1)` CMACs `+ 1 IFFT(2B)` per `B`-sample block (`P = ceil(L / B)`).
   - `HybridConvolver (BURST)`: `O(B_host · B_tail)` direct work every callback + 1 `PartitionedConvolver(B_tail)` every `B_tail / B_host` callbacks.
   - `HybridConvolver (SPREAD)`: `O(B_host · B_tail)` direct work + `ceil((P - 1) / (B_tail / B_host - 1))` past-partition CMACs on pre-completion callbacks, and `RFFT + 1 CMAC(p=0) + IFFT` on the completion callback.
   - `NonUniformConvolver` & `FixedQ15NonUniformConvolver` (time-distributed, Issue #16): each FDL stage subdivides its block into `U = rfft_steps + P + ifft_steps` atomic work units (`log2(B)` steps/transform in `SimdFixedFftBackend`, `log2(2B)` in `fft_q15`, `1` in monolithic backends) distributed via the periodic ceiling schedule `target(slot) = ceil((slot + 1) * U / span)` over `span = min(slack_s, B_s - 1) + 1` samples with deferral `defer_s = slack_s.saturating_sub(B_s - 1)`. Any contiguous `H`-sample callback (`H = q * B_s + rem`, regardless of alignment) executes at most `q * U + min(U, ceil(rem * U / span))` units, with zero integer division in the per-sample loop (power-of-two `2 * B_s` output delay rings).
5. **State transition safety**: `HybridConvolver::set_tail_spreading()` returns `Err(ConvError::InvalidStateTransition)` if called mid-cycle (`tail_pos != 0 || acc_primed`). Out-of-range staged transform steps likewise return `Err(ConvError::InvalidStateTransition)`.
6. **Panic freedom**: All buffer lengths, indices, and workspace sizes are validated via typed `ConvError` returns; `#![forbid(unsafe_code)]` is enforced at crate root.

## IEEE 754 Subnormal Guard

`f32` engines flush every magnitude below `simd_kernels::DENORMAL_GUARD = 2^-63` (≈ `1.08e-19`, ≈ −379 dBFS) to zero, at these points:

- input samples;
- IR coefficients and their partition spectra (at `prepare`);
- each FDL input slot after its forward FFT completes;
- the spectral accumulators before the inverse FFT starts;
- the output samples.

Guarding both inputs and coefficients at `2^-63` means every non-zero product is `≥ 2^-126`, a normal number. The slot, accumulator and output flushes remove subnormal residues left by cancellation. A decaying input therefore drains to exact zeros (`tests/denormal_guard.rs`, covering all sizes up to `N16384` and time-distributed Strategy-1 NUPC).

Without the guard, on x86-64 without FTZ/DAZ, a 1024-tap `DirectConvolver` took 346 µs per 32-sample block for 1e-37 inputs instead of 4.3 µs, which is 52 % of a 666 µs budget. Hosts may still enable FTZ/DAZ in their audio thread for complete coverage. The crate cannot do it itself under `#![forbid(unsafe_code)]`. Q1.15 engines are integer-only and unaffected.

## Admission-Control Model (`NonUniformBudget`)

`NonUniformPlan::estimated_worst_ops_per_callback` (and backend-specific `estimated_worst_ops_for_backend` / `check_budget_for_backend`) is an alignment-independent **operation count**, not a cycle count. For any contiguous host callback of `H` samples (`H = q * B_s + rem`) it sums:

- the direct-head MACs (`L_0 · H`);
- for each FDL stage, `q` full blocks (`2 * transform_ops + P * 4 * (B_s + 1)`, with `transform_ops = 5 * N * log2 N`) plus the worst-case operation cost of `u_rem = min(U, ceil(rem * U / span))` subdivided units (assigning highest-cost transform steps before CMAC partitions).

Deterministic per-sample op-count traces (`tests/time_distributed_fft.rs`, using `stage_work_item_ops`) verify that every sliding `H`-sample window across all alignments satisfies `actual_callback_ops <= estimated_worst_ops_per_callback()` (and `<= estimated_worst_ops_for_backend(&fft)` for monolithic backends such as `MicroFftBackend`), achieving `peak/mean = 1.21×–1.76×` on Strategy-1 plans (`4k`–`64k` taps, `H = 64`).

Memory admission for `FixedQ15NonUniformConvolver` includes both workspace bytes (shared Q1.15 input ring, power-of-two `2 * B_s` Q1.23 output rings, per-stage `i64` accumulators also serving as in-place FFT buffers) and `instance_bytes` (`< 4 KiB`).

## Empirical Measurements vs. Formal WCET

Host benchmarks (`HOST_ONLY`) measure GUM Type A mean ± expanded uncertainty (`k = 2`), median (`p50`), `p95`, `p99`, and observed maximum. Observed maxima are **not** formal WCET proofs (`TARGET_RT_VALIDATED = OPEN`). Formal hard-RT certification requires target-specific static timing analysis, cache/ISR modeling, and hardware-in-the-loop stress testing.
