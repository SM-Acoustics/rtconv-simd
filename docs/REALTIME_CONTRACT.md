# Real-Time Contract (`docs/REALTIME_CONTRACT.md`)

Aligned with `SM-ENG-002` (*Proposed v1*): `prepare()` belongs to **Class B** (off-callback bounded preparation); `process_sample()`, `process_block()`, and `reset()` belong to **Class A** (hard-deadline audio processing).

## Class A Invariants (`PROVEN`, `HOST_ONLY`)

1. **Zero heap activity**: Verified via serialized `GlobalAlloc` (`tests/no_alloc_rt.rs`) for `DirectConvolver`, `PartitionedConvolver`, `HybridConvolver` (`BURST` and `SPREAD`), `MicroFftBackend`, `SimdFixedFftBackend`, and `PhastFftBackend` (with pre-initialized planner).
2. **Zero blocking operations**: No mutexes, channels, threads, syscalls, file I/O, or logging in `src/`.
3. **Zero runtime CPUID or FFT planning in callback**: `SimdBackend` and `PhastFftBackend` planners are resolved once prior to streaming.
4. **Bounded per-callback work**:
   - `DirectConvolver`: `O(B · L)` multiply-adds per block of `B` samples and IR length `L`.
   - `PartitionedConvolver`: `1 RFFT(2B) + P · (B + 1)` CMACs `+ 1 IFFT(2B)` per `B`-sample block (`P = ceil(L / B)`).
   - `HybridConvolver (BURST)`: `O(B_host · B_tail)` direct work every callback + 1 `PartitionedConvolver(B_tail)` every `B_tail / B_host` callbacks.
   - `HybridConvolver (SPREAD)`: `O(B_host · B_tail)` direct work + `ceil((P - 1) / (B_tail / B_host - 1))` past-partition CMACs on pre-completion callbacks, and `RFFT + 1 CMAC(p=0) + IFFT` on the completion callback.
5. **State transition safety**: `HybridConvolver::set_tail_spreading()` returns `Err(ConvError::InvalidStateTransition)` if called mid-cycle (`tail_pos != 0 || acc_primed`).
6. **Panic freedom**: All buffer lengths, indices, and workspace sizes are validated via typed `ConvError` returns; `#![forbid(unsafe_code)]` is enforced at crate root.

## Admission-Control Model (`NonUniformBudget`)

`NonUniformPlan::estimated_worst_ops_per_callback` is a conservative **operation count**, not a cycle count. For a host callback of `H` samples it includes:

- the direct-head MACs (`L_0 · H`);
- for each FDL stage, `ceil(H / B)` block triggers, each counted as an RFFT and an IFFT at `5·N·log2 N` operations, plus `4·(B+1)` real MACs per partition;
- for SPREAD stages, one extra in-flight completion and the CMAC sub-steps that can fall inside the callback.

Memory admission for `FixedQ15NonUniformConvolver` includes `instance_bytes` (the in-struct `i64` accumulators).

## Empirical Measurements vs. Formal WCET

Host benchmarks on Windows 11 Pro (`HOST_ONLY`) measure median, p95, p99, and observed maximum. Observed maxima are **not** formal WCET proofs (`TARGET_RT_VALIDATED = OPEN`). Formal hard-RT certification requires target-specific static timing analysis, cache/ISR modeling, and hardware-in-the-loop stress testing.
