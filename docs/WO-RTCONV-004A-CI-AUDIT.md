# WO-RTCONV-004A — CI, macOS Zero-Allocation Harness & Dependabot Audit

**Work Order**: `RTCONV-004A` (Priority P0)  
**Branch**: `fix/rtconv-004a-ci-harness` (branched from `origin/main` at `e214f74`)  
**Date**: `2026-11-23`  
**Registry State Checked**: GitHub Release `v0.1.0` (`0e17e20`, published `2026-11-22T23:57:17Z`); `crates.io` `rtconv-simd` returns HTTP `404` (not yet published to crates.io).

---

## 1. Root-Cause Analysis of CI Failures

### 1.1 Intermittent macOS `no_alloc_rt` Failure (Run `45086089406`, PR #3 Run `45085775516`)

- **Observed Symptom**: On `macos-latest` (`aarch64-apple-darwin`), `test_zero_heap_allocations_all_engines_and_backends` failed intermittently with 5 allocations detected on `main` (`45086089406`) and 3 allocations detected on Dependabot PR #3 (`45085775516`), while passing on PR #2 (`45085770829`) and PR #4 (`45094250013`).
- **Root Cause**:
  1. `tests/no_alloc_rt.rs` executed under Rust's default `libtest` harness (`harness = true`). Even with a single `#[test]` function, `libtest` spawns a worker thread while the main harness thread stays active (polling the result channel and managing output capture).
  2. `CountingAllocator` gated counting on a process-wide `GUARD_ACTIVE: AtomicBool`. On macOS, concurrent `libtest` harness / OS runtime allocations occurring on the non-worker thread while `GUARD_ACTIVE == true` were falsely attributed to `rtconv_simd`.
  3. Attempting to filter by `thread_local!` inside `GlobalAlloc::alloc` is unsafe across Rust versions: on Rust `1.89.0` (`x86_64-pc-windows-gnu`), first-access TLS initialization calls `System.alloc`, causing infinite recursion (`STATUS_STACK_OVERFLOW`).

### 1.2 Dependabot MSRV Corruption (PR #3)

- **Observed Symptom**: Dependabot PR #3 (`https://github.com/SM-Acoustics/rtconv-simd/pull/3`, commit `d1f9f2f`) replaced `dtolnay/rust-toolchain@1.89.0` with `dtolnay/rust-toolchain@1.120.0`, causing `rustup` to fail with HTTP `404` on the `MSRV (Rust 1.89.0)` job (`45085775516`).
- **Root Cause**: Using a Rust version string (`@1.89.0`) as the Git ref of `dtolnay/rust-toolchain` causes Dependabot's `github-actions` ecosystem to treat the toolchain version as an Action semver tag and auto-bump it to pre-created future tags (`@1.120.0`).

### 1.3 Review of Open Dependabot PR #2

- **PR #2** (`https://github.com/SM-Acoustics/rtconv-simd/pull/2`, commit `1a7595c`): `chore(deps): bump actions/checkout from 4 to 7`.
- **Status**: All checks passed (`45085770829`, `45085770855`), but it remains `OPEN` pending maintainer review and is preserved untouched (no auto-merge).
- **PR #3**: Must **not** be merged because it corrupts the MSRV toolchain declaration; superseded by the SHA-pinned workflow and Dependabot ignore policy in this PR.

---

## 2. Implemented Corrections

### 2.1 Standalone Zero-Allocation Harness (`tests/no_alloc_rt.rs`, `harness = false`)

- Declared `[[test]] name = "no_alloc_rt" harness = false` in `Cargo.toml`.
- Eliminates `libtest` worker/runner multithreading completely: a single `fn main()` thread executes in the process.
- Environment warm-up (`SimdBackend::detect_or_baseline()` and initial stdout) runs **before** arming `GUARD_ACTIVE`.
- **Positive Control**: `verify_positive_control()` performs a deliberate `black_box(Box::new([0xA5u64; 16]))` allocation and `drop` while `GUARD_ACTIVE` is armed and asserts `allocs >= 1 && deallocs >= 1`, proving the instrumentation actively catches heap allocations.
- **Separated Phase Measurements**: Independent `assert_zero_alloc(...)` checks for:
  - `prepare` (`DirectConvolver`, `PartitionedConvolver`, `HybridConvolver` with `MicroFftBackend` and `SimdFixedFftBackend`)
  - `process_sample` (`DirectConvolver`)
  - `process_block` (`Direct`, `Partitioned`, `Hybrid BURST`, `Hybrid SPREAD`, and `#[cfg(feature = "phastft")] PhastFftBackend`)
  - `reset` (all engines and backends)
  - `metadata` (`latency_samples()`, `capabilities()`)
- **Safety & Limitations**:
  - `unsafe impl GlobalAlloc` lives strictly inside `tests/no_alloc_rt.rs` (zero `unsafe` in `src/`).
  - Uses only lock-free `AtomicBool` / `AtomicUsize` (zero TLS inside `alloc`/`dealloc`, 100% compatible with Rust `1.89.0`).
  - No `#[ignore]`, no macOS conditionals, no `continue-on-error`.
  - *Limitation*: Intercepts Rust `GlobalAlloc` only (sufficient because `#![no_std]` `#![forbid(unsafe_code)]` core code cannot invoke direct OS allocators or FFI).

### 2.2 SHA-Pinned `dtolnay/rust-toolchain` & Dependabot Policy

- Pinned `dtolnay/rust-toolchain@df2a88e9fe4098bad09cb42c3dec7187f15861fc` (`master` commit verified via GitHub API) in `.github/workflows/ci.yml`, `embedded.yml`, and `release-check.yml`, passing `toolchain: "1.89.0"` or `toolchain: stable` explicitly via `with:`.
- Added an explicit `ignore` rule for `dtolnay/rust-toolchain` in `.github/dependabot.yml` so Dependabot never auto-bumps the MSRV toolchain.

---

## 3. Verification Evidence

- **Local Windows (`rustc 1.100.0`, `x86_64-pc-windows-gnu`) & MSRV (`rustc 1.89.0`)**:
  - `cargo fmt --all -- --check`: `PASS`
  - `cargo clippy --lib --no-default-features -- -D warnings`: `PASS`
  - `cargo clippy --all-targets --all-features -- -D warnings`: `PASS`
  - `cargo +1.89.0 test --no-default-features` & `--all-features`: `PASS`
  - `cargo test --release --no-default-features` & `--all-features`: `PASS`
  - `cargo test --release --all-features --test no_alloc_rt`: `PASS` (positive control + all zero-alloc phases)
  - `python3 scripts/agentic_guard.py --base main`: `PASS` (0 blocked)
- **Remote CI (`SM-Acoustics/rtconv-simd` PR for `fix/rtconv-004a-ci-harness`)**: Recorded in PR checks (Linux, Windows, macOS, MSRV `1.89.0`, Embedded cross-targets, Agentic Software Assurance).
