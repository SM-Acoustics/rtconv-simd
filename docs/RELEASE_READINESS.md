# Release Readiness — `rtconv-simd` v0.1.0

**Release Status (2026-11-22)**: `GITHUB_RELEASE_PUBLISHED / CRATES_IO_PENDING_OWNER_AUTH`. The public repository and GitHub release [`v0.1.0`](https://github.com/SM-Acoustics/rtconv-simd/releases/tag/v0.1.0) were created after all CI, MSRV, embedded cross-compilation, and release dry-run gates passed. The first crates.io upload has **not** occurred because registry credentials are not configured. `DEVICE_VERIFIED` remains `NO`.

## 1. Verification Dimensions (SM-ENG-004 *Proposed v1*)

| Dimension | Status | Details |
| :--- | :--- | :--- |
| **`MSRV_VERIFIED`** | **PASS (`PROVEN`)** | `rustc 1.89.0 (29483883e 2025-08-04)`, host `x86_64-pc-windows-gnu`: `cargo +1.89.0 check --lib --no-default-features`, `cargo +1.89.0 check --all-features`, `cargo +1.89.0 test --no-default-features` (21/21), and `cargo +1.89.0 test --all-features` (22/22) all `EXIT 0`. Enforced in CI via `ci.yml` (`msrv`) and `release-check.yml` (`msrv-check`). |
| **`HOST_TESTED`** | **PASS (`PROVEN`)** | `SalatielPC` (AMD Ryzen 3 PRO 2300U, Windows 11 Pro `10.0.26300`): `rustc 1.100.0 (0ef1bc386 2026-11-09)`, **`host: x86_64-pc-windows-gnu`** (confirmed verbatim via `rustc -vV` and `rustup show`: only `stable-x86_64-pc-windows-gnu` and `1.89.0-x86_64-pc-windows-gnu` are installed on this host). GitHub Actions `ci.yml` validated `ubuntu-latest`, `macos-latest` (`aarch64-apple-darwin`), and `windows-latest` (`x86_64-pc-windows-msvc`): **PASS**. |
| **`CROSS_COMPILE_VERIFIED`** | **PASS (`PROVEN`, compile-only)** | `.github/workflows/embedded.yml` passed for `thumbv7em-none-eabihf`, `riscv32imac-unknown-none-elf`, `aarch64-unknown-none`, and `wasm32-unknown-unknown`. No device execution claimed. |
| **`DEVICE_VERIFIED`** | **`NO` (`OPEN`)** | No physical MCU/SBC/ESP32-P4 hardware execution has been performed. |

## 2. Release Identification & Provenance

| Field | Value |
| :--- | :--- |
| **Project / Package / Lib** | `RTConv SIMD` / `rtconv-simd` / `rtconv_simd` (`0.1.0`, `MIT OR Apache-2.0`) |
| **GitHub Repository** | Public: https://github.com/SM-Acoustics/rtconv-simd ; tag [`v0.1.0`](https://github.com/SM-Acoustics/rtconv-simd/releases/tag/v0.1.0), merge `0e17e20513fd10f96da854d6e2089350d871a3b6` |
| **Branch & Preserved History** | `release/0.1.0` over `main` (`fd79330` -> `899d4b2` -> `2d145ad` -> `c722c4f` -> WO-RTCONV-003.1 commit) |
| **Verification Date** | `2026-11-22` |
| **Host `rustc -vV` (verbatim)** | `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `host: x86_64-pc-windows-gnu`, `LLVM version: 23.1.5` |
| **MSRV `rustc -vV` (verbatim)** | `rustc 1.89.0 (29483883e 2025-08-04)`, `host: x86_64-pc-windows-gnu` (`edition = "2024"`, `rust-version = "1.89"`) |
| **SM-Acoustics Standards** | `SM-ENG-001`..`005`, `GOVERNANCE.md`, `SECURITY.md`, `CONTRIBUTING.md` (`https://github.com/SM-Acoustics/.github`, all **Proposed v1**, 2026-11-13) |

---

## 3. Gate-by-Gate Evidence Matrix (G0–G13)

| Gate | Requirement | Status | Evidence / Exact Command & Result |
| :--- | :--- | :--- | :--- |
| **G0** | Clean Git baseline & branch hygiene | **PASS** | `fd79330`, `899d4b2`, `2d145ad`, `c722c4f` preserved intact; clean local commit on `release/0.1.0`; independent of private `SalatielBM22/RTconv`. |
| **G1** | GitHub repository and Cargo metadata | **PASS (GitHub)** | Public repository created in `SM-Acoustics`; `repository` and `documentation` are set in `Cargo.toml` (commit `20b912f`). Registry publication remains pending owner authentication. |
| **G2** | Pure `#![no_std]`, zero `alloc`, `#![forbid(unsafe_code)]`, `default = []`, MSRV `1.89.0` | **PASS** | `cargo +1.89.0 check --lib --no-default-features` (`EXIT 0`), `cargo +1.89.0 check --all-features` (`EXIT 0`), `cargo check --lib --no-default-features` (`EXIT 0`), `cargo check --all-features` (`EXIT 0`). |
| **G3** | Formatting & Clippy zero-warnings | **PASS** | `cargo fmt --all -- --check`, `cargo clippy --lib --no-default-features -- -D warnings`, `cargo clippy --all-targets --all-features -- -D warnings`: all `EXIT 0`. |
| **G4** | Full numerical, streaming, boundary & state-transition suite | **PASS** | Stable `1.100.0` (`--release`): `--no-default-features` **21/21 ok**, `--all-features` **22/22 ok**. MSRV `1.89.0`: `--no-default-features` **21/21 ok**, `--all-features` **22/22 ok**. *(Test count reflects consolidating `tests/no_alloc_rt.rs` into one sequential entry point covering all 3 backend scenarios without TLS recursion on Rust 1.89.0.)* |
| **G5** | Zero heap allocations in `prepare` & callbacks | **PASS** | `cargo test --release --all-features --test no_alloc_rt -- --test-threads=1` (`EXIT 0`) and under default parallel execution on both `1.89.0` and `1.100.0`. Verifies 0 allocs / 0 deallocs across `MicroFftBackend`, `SimdFixedFftBackend` (`SPREAD`), and `PhastFftBackend` callbacks. |
| **G6** | Runnable public examples | **PASS** | `direct_fir`, `hybrid_convolution`, `backend_selection --all-features`, `multiple_instances`: all `EXIT 0`. |
| **G7** | Public API docs without warnings | **PASS** | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`: `EXIT 0`. |
| **G8** | Performance & SIMD evidence without overclaiming | **PASS** | WO-002 benchmarks & disassembly (`FMA=96`, `YMM=527`, `XMM=5251`) documented in `docs/BENCHMARKS.md` and `docs/SIMD_BACKENDS.md` under `HOST_TESTED` (`x86_64-pc-windows-gnu`, Ryzen 3 PRO 2300U). Disclaims `HARD_RT_CERTIFIED` and RISC-V RVV SIMD. |
| **G9** | License & supply-chain audit (`SM-ENG-005` Proposed v1) | **PASS** | `LICENSE-MIT` + `LICENSE-APACHE` packaged; 100% permissive licenses (`MIT`, `Apache-2.0`, `Unicode-3.0`), zero C/C++/FFI, `phastft` with `default-features = false` (no `rayon`). |
| **G10** | Clean `.crate` packaging & dry-run (no `--allow-dirty`) | **PASS** | `cargo package --list`, `cargo package`, and `cargo publish --dry-run`: **45 files** packaged cleanly (`EXIT 0`). Cargo metadata URLs are populated and the online `cargo publish --dry-run` completed with exit code 0. |
| **G11** | Independent consumer crate validation | **PASS** | Fresh consumer crate depending on unpacked `target/package/rtconv-simd-0.1.0` compiled and ran `Direct`, `Partitioned` (`SimdFixedFftBackend`), and `Hybrid` (`SPREAD`) in `--release`: `CONSUMER_OK ... simd_active=true` (`EXIT 0`). |
| **G12** | Remote CI (`ci.yml`, `embedded.yml`, `release-check.yml`) | **PASS** | PR CI, MSRV `1.89.0`, three desktop OS jobs, four `no_std` cross-target jobs, and manual `release-check.yml` completed successfully on GitHub Actions. See run `45085607229`. |
| **G13** | Owner approval, GitHub publication & `crates.io` publish | **PARTIAL (registry pending)** | Owner authorized repository publication. PR #1 merged; GitHub repository made public; tag and release `v0.1.0` published with the `.crate` artifact. `cargo publish` has **not** been executed because crates.io registry credentials are absent. |
