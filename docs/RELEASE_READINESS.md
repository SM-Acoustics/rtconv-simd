# Release Readiness — `rtconv-simd` v0.1.0

**Release Candidate Status**: `RELEASE_BLOCKED / EVIDENCE_REQUIRED` (Local RC technical & packaging gates passed; blocked solely on external GitHub repository creation `SM-Acoustics/rtconv-simd`, uncommenting `repository` in `Cargo.toml`, remote CI execution, and explicit owner approval before `crates.io` publication.)

## 1. Release Identification & Provenance

| Field | Value |
| :--- | :--- |
| **Project** | RTConv SIMD |
| **Cargo Package** | `rtconv-simd` (`[lib] name = "rtconv_simd"`) |
| **Target Version** | `0.1.0` |
| **License** | `MIT OR Apache-2.0` |
| **Intended GitHub Repository** | `SM-Acoustics/rtconv-simd` *(not yet created — HTTP 404 as of 2026-11-22)* |
| **Working Branch** | `release/0.1.0` (branched from `main` at `899d4b2`; WO-001 baseline `fd79330`) |
| **RC Verification Commit** | `2d145ad` *(followed by this `docs/RELEASE_READINESS.md` commit)* |
| **Verification Date** | `2026-11-22` |
| **Verification Scope** | `HOST_ONLY` (no physical embedded MCU/SBC hardware execution yet) |
| **Host Hardware** | AMD Ryzen 3 PRO 2300U w/ Radeon Vega Mobile Gfx (4C/4T, x86-64 Zen 1; SSE2, SSE4.2, AVX2, FMA; no AVX-512) |
| **Host OS** | Microsoft Windows 11 Pro, Version `10.0.26300` (64-bit) |
| **Toolchain** | `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0 (3acc79c0c 2026-11-05)` |
| **Host Target** | `x86_64-pc-windows-gnu` |
| **MSRV Declared** | `rust-version = "1.89"`, `edition = "2024"` |
| **SM-Acoustics Standards Referenced** | `SM-ENG-001`..`005`, `GOVERNANCE.md`, `SECURITY.md`, `CONTRIBUTING.md` (`https://github.com/SM-Acoustics/.github`, all **Proposed v1**, 2026-11-13) |

---

## 2. Gate-by-Gate Evidence Matrix (G0–G13)

| Gate | Requirement | Status | Confidence | Scope | Evidence / Exact Command & Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **G0** | Clean Git baseline & branch hygiene (`main` + `release/0.1.0`) | **PASS** | `PROVEN` | `HOST_ONLY` | Baseline `fd79330` (WO-001) & `899d4b2` (WO-002) preserved on `main`; RC committed cleanly on `release/0.1.0` (`2d145ad`, zero untracked/dirty files). Independent of private `SalatielBM22/RTconv`. |
| **G1** | Name availability on `crates.io` & GitHub repo existence | **PARTIAL (BLOCKED)** | `PROVEN` | `HOST_ONLY` | `crates.io`: `rtconv-simd` and `rtconv_simd` both HTTP 404 (available). **Blocker**: `SM-Acoustics/rtconv-simd` returns HTTP 404 and `gh` CLI is not installed on `SalatielPC`; `repository` remains commented out in `Cargo.toml` until repo exists. |
| **G2** | Pure `#![no_std]`, zero `std`/`alloc` in core, `default = []`, `#![forbid(unsafe_code)]` | **PASS** | `PROVEN` | `HOST_ONLY` | `cargo check --lib --no-default-features` -> `EXIT 0`; `cargo check --all-features` -> `EXIT 0`. `#![no_std]` and `#![forbid(unsafe_code)]` enforced in `src/lib.rs`. `cargo tree --no-default-features -e normal` shows only `fearless_simd 1.2.0` (`libm 0.2.16`), `fearless_simd_macros 0.1.0`, `microfft 0.6.0`. |
| **G3** | Formatting & Clippy zero-warnings across all targets | **PASS** | `PROVEN` | `HOST_ONLY` | `cargo fmt --all -- --check` -> `EXIT 0`; `cargo clippy --lib --no-default-features -- -D warnings` -> `EXIT 0`; `cargo clippy --all-targets --all-features -- -D warnings` -> `EXIT 0`. |
| **G4** | Full numerical, streaming, boundary & state-transition test suite | **PASS** | `PROVEN` | `HOST_ONLY` | `cargo test --release --no-default-features` -> `22 passed, 0 failed` (`EXIT 0`); `cargo test --release --all-features` -> `24 passed, 0 failed` (`EXIT 0`). Covers `HybridConvolver::set_tail_spreading` mid-cycle `InvalidStateTransition`, arbitrary/partial blocks, denormals, stereo isolation, checked overflow, and cross-backend equivalence (`MicroFft`, `SimdFixedFft`, `PhastFft`). |
| **G5** | Zero heap allocations in `prepare` and audio callbacks | **PASS** | `PROVEN` | `HOST_ONLY` | `cargo test --release --all-features --test no_alloc_rt -- --test-threads=1` -> `3 passed, 0 failed` (`EXIT 0`); also passes under default parallel test execution via `Mutex<()>` + thread-local `IN_GUARD`. Verifies 0 allocs / 0 deallocs for `MicroFftBackend`, `SimdFixedFftBackend` (`SPREAD`), and `PhastFftBackend` callbacks. |
| **G6** | Runnable public examples across core and optional features | **PASS** | `PROVEN` | `HOST_ONLY` | `cargo run --example direct_fir`, `hybrid_convolution`, `backend_selection --all-features`, `multiple_instances` -> all `EXIT 0`. |
| **G7** | Public API documentation builds cleanly without warnings | **PASS** | `PROVEN` | `HOST_ONLY` | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` -> `EXIT 0`. Complete `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `docs/*.md` authored in natural technical English. |
| **G8** | Performance & SIMD evidence documented without overclaiming | **PASS** | `PROVEN` | `HOST_ONLY` | WO-002 benchmarks & disassembly (`FMA=96`, `YMM=527`, `XMM=5251`; `SimdFixed(SSE2)` `2.08x–2.39x` isolated FFT speedup; `Hybrid SPREAD` p99 `0.78%` of `666.67 µs` budget at 16,383 taps) documented in `docs/BENCHMARKS.md` and `docs/SIMD_BACKENDS.md`. Explicitly disclaims `HARD_RT_CERTIFIED` and RISC-V RVV SIMD. |
| **G9** | License & supply-chain audit (`SM-ENG-005` Proposed v1) | **PASS** | `PROVEN` | `HOST_ONLY` | Dual-licensed `MIT OR Apache-2.0` (`LICENSE-MIT` + `LICENSE-APACHE` packaged). `cargo tree --no-default-features -e normal -f "{p} ({l})"` and `--all-features` confirm 100% permissive licenses (`MIT`, `Apache-2.0`, `Unicode-3.0`), zero C/C++/FFI, zero copyleft, `phastft` configured with `default-features = false` (no `rayon`). |
| **G10** | Clean `.crate` packaging & `crates.io` dry-run (no `--allow-dirty`) | **PASS** | `PROVEN` | `HOST_ONLY` | `cargo package --list` -> `EXIT 0` (44 files); `cargo package` -> `EXIT 0` (`235.7 KiB` / `54.0 KiB` compressed); `cargo publish --dry-run` -> `EXIT 0` (aborted due to dry run). Note: expected Cargo warning `manifest has no documentation, homepage or repository` until G1 is resolved. |
| **G11** | Independent consumer crate validation from unpacked `.crate` | **PASS** | `PROVEN` | `HOST_ONLY` | Fresh temporary consumer crate referencing `target/package/rtconv-simd-0.1.0` with default (`no_std`-compatible) features compiled and executed `DirectConvolver`, `PartitionedConvolver` (`SimdFixedFftBackend`), and `HybridConvolver` (`prepare_with_tail_spreading`) in `--release` -> `CONSUMER_OK ... simd_active=true` (`EXIT 0`). |
| **G12** | Multi-OS / MSRV / embedded cross-target CI verification | **OPEN** | `INFERRED` | `HOST_ONLY` | `.github/workflows/{ci.yml, embedded.yml, release-check.yml}` authored covering Linux/macOS/Windows, MSRV `1.89`, `thumbv7em-none-eabihf`, `aarch64-unknown-none`, `riscv32imac-unknown-none-elf`, and `wasm32-unknown-unknown`. Not yet executed on GitHub Actions because `SM-Acoustics/rtconv-simd` does not exist yet. |
| **G13** | Owner approval, GitHub publication & `crates.io` publish | **OPEN** | `OPEN` | `HOST_ONLY` | Intentionally held per WO-RTCONV-003. No `git push`, no public visibility toggle, and no real `cargo publish` executed. |

---

## 3. Open Blockers & Required Owner Actions Before `0.1.0` Publish

1. **Create Private GitHub Repository (`SM-Acoustics/rtconv-simd`)**:
   - Create `SM-Acoustics/rtconv-simd` as a **private** repository under the `SM-Acoustics` organization (either via GitHub Web UI or after installing GitHub CLI `gh`).
   - Add topics: `rust, dsp, no-std, simd, embedded, fft, convolution, real-time, audio, fir-filter`.
2. **Uncomment Metadata in `Cargo.toml`**:
   - Uncomment `repository = "https://github.com/SM-Acoustics/rtconv-simd"` and `documentation = "https://docs.rs/rtconv-simd"` in `Cargo.toml`, commit cleanly on `release/0.1.0`, and re-run `cargo package` and `cargo publish --dry-run` (verifying zero Cargo metadata warnings).
3. **Push & Run GitHub Actions CI (G12)**:
   - Add remote `origin` pointing to `https://github.com/SM-Acoustics/rtconv-simd.git`, push `main` and `release/0.1.0`, open PR `release/0.1.0 -> main`, and verify green runs on `ci.yml`, `embedded.yml` (cross-target `no_std` builds), and `release-check.yml` (including MSRV `1.89`).
4. **Final Owner Approval & Release (G13)**:
   - Merge PR to `main`, switch repository visibility to **public** when ready, tag `v0.1.0`, and execute `cargo publish` only after explicit owner sign-off.
