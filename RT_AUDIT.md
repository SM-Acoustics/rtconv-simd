# Real-Time Invariants & Panic/Allocation Audit (`RT_AUDIT.md`) — WO-RTCONV-002

## 1. Real-Time Design Invariants (`RT_DESIGN_PASS`)

| Invariant | Status | Verification |
| :--- | :--- | :--- |
| **`#![no_std]` core** | Verified | `#![no_std]` in `src/lib.rs`; `cargo check --lib --no-default-features` exits `0`. Both `MicroFftBackend` and `SimdFixedFftBackend` compile with zero `std` and zero `alloc`. |
| **Zero `alloc` in core** | Verified | `cargo tree --no-default-features -e normal` contains only `fearless_simd 1.2.0` (`libm 0.2.16`), `fearless_simd_macros 0.1.0` (proc-macro), and `microfft 0.6.0`. |
| **Zero heap allocation in audio callback (all 3 FFT backends)** | Verified | `tests/no_alloc_rt.rs` (`CountingAllocator`) asserts `0` allocations and `0` deallocations across `MicroFftBackend`, `SimdFixedFftBackend`, `PhastFftBackend` (pre-initialized planner), and `HybridConvolver` (`tail_spreading = true`). |
| **Zero `unsafe` in crate** | Verified | `#![forbid(unsafe_code)]` at crate root (`src/lib.rs`). |
| **Zero locks / threads / I/O in callback** | Verified | `phastft` uses `default-features = false` (no `rayon`) and sets `multithreaded_bit_reversal = false`, `smallest_parallel_chunk_size = usize::MAX`. |
| **Pre-validated SIMD & FFT plans** | Verified | `SimdBackend` and `PhastFftBackend` planners are created once during initialization; `SimdFixedFftBackend` uses `const` twiddle tables. |

## 2. Panic & Transitive Allocation Audit (Audio Path)

- **Core crate (`src/`)**:
  - Zero explicit `panic!`, `unwrap()`, or `expect()` calls in `src/direct.rs`, `src/partitioned.rs`, `src/hybrid.rs`, `src/fft.rs`, `src/simd_fft.rs`, `src/phastft_backend.rs`, and `src/simd_kernels.rs`.
  - Buffer dimensions are validated at entry (`MismatchedBufferLength`, `WorkspaceTooSmall`) before indexing.
  - Circular buffer and FDL ring indices are bounded by `ir_len` / `num_partitions` validated in `prepare()`.
- **Transitive dependencies in audio path**:
  - `microfft 0.6.0`: `rfft_*` internally calls `.try_into().unwrap()` on fixed-size arrays of exact compile-time length `N/2` (provably infallible when `input: &mut [f32; N]`); zero heap allocations.
  - `phastft 0.4.1` (`--features phastft` only): `r2c_fft_f32_with_planner_and_opts` and `c2r_fft_f32_with_planner_and_opts` contain `assert_eq!` length checks on input/output/scratch slices; `PhastFftBackend` validates all exact lengths (`n`, `half + 1`, `half`) beforehand and returns `Err(ConvError)` so those asserts are unreachable on valid calls. `PlannerR2c32::new()` allocates `Vec<f32>` twiddles **only at setup time**, never in the callback (verified by `test_zero_heap_allocations_phastft_audio_callback`).
  - `fearless_simd 1.2.0`: `from_slice`/`store_slice` panic only if slice length `< lanes`; all callers guard loop bounds with `i + lanes <= n` or exact power-of-two `half % lanes == 0`.

## 3. Stack & Memory Footprint Comparison

- **`MicroFftBackend`**: `0 B` external scratch, but allocates `[Complex32; N]` on the stack inside `rfft_inverse` (`512 B` for `N=64`, `1,024 B` for `N=128`, `2,048 B` for `N=256`).
- **`SimdFixedFftBackend`**: Uses external caller-provided scratch (`256 B` for `N=64`, `512 B` for `N=128`, `1,024 B` for `N=256`). Peak stack frame is `< 256 B`, making it safer for tight embedded ISR stacks.
- **`PhastFftBackend`**: Uses external caller-provided scratch (`256 / 512 / 1,024 B`) plus heap-allocated `PlannerR2c32` twiddles created once before streaming (desktop only).

## 4. Real-Time Timing Boundaries & Limitations

- **Algorithmic Latency**: `DirectConvolver = 0`, `PartitionedConvolver = 0` internal extra samples on `B`-aligned blocks, `HybridConvolver = 0` (both `BURST` and `SPREAD` modes).
- **Tail Spreading Trade-off**: `set_tail_spreading(true)` eliminates the large phase-3 spike on long IRs (`16,383` taps: `16.90 µs → 5.20 µs` p99 on `SimdFixedFft(SSE2)`), while modestly raising phases 0–2 from `~1.0 µs` to `~3.2 µs` median. For short IRs (`≤ 2,048` taps) with `MicroFftBackend`, burst mode can remain lower-jitter.
- **No Formal WCET Certification**: Empirical Windows 11 Pro measurements (median, p99, max observed) are **not** `HARD_RT_CERTIFIED` WCET proofs. Target-specific static timing analysis and RTOS/bare-metal validation are required for certification.
- **Architecture Status**: x86-64 (Scalar, SSE2, SSE4.2, AVX2+FMA) verified on `AMD Ryzen 3 PRO 2300U`. AVX-512, ARM64 NEON, and WASM SIMD128 are supported by `fearless_simd 1.2.0` but `NOT_VERIFIED` on hardware here. RISC-V uses the explicit scalar fallback (`is_simd_accelerated() == false`); no RVV instructions are claimed.
