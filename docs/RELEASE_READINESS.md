# Release Readiness — `rtconv-simd` v0.1.0

**Release Candidate Status**: `RELEASE_BLOCKED / EVIDENCE_REQUIRED` (Local technical, MSRV `1.89.0`, and packaging gates **G0, G2–G11 PASS**; **G1** blocked on `SM-Acoustics/rtconv-simd` creation & `Cargo.toml` metadata uncommenting, **G12** `PENDING_REMOTE_CI`, **G13** `AWAITING_OWNER_APPROVAL`.)

## 1. Verification Dimensions (SM-ENG-004 *Proposed v1*)

| Dimension | Status | Details |
| :--- | :--- | :--- |
| **`MSRV_VERIFIED`** | **PASS (`PROVEN`)** | `rustc 1.89.0 (29483883e 2025-08-04)`, host `x86_64-pc-windows-gnu`: `cargo +1.89.0 check --lib --no-default-features`, `cargo +1.89.0 check --all-features`, `cargo +1.89.0 test --no-default-features` (21/21), and `cargo +1.89.0 test --all-features` (22/22) all `EXIT 0`. Enforced in CI via `ci.yml` (`msrv`) and `release-check.yml` (`msrv-check`). |
| **`HOST_TESTED`** | **PASS (`PROVEN`)** | `SalatielPC` (AMD Ryzen 3 PRO 2300U, Windows 11 Pro `10.0.26300`): `rustc 1.100.0 (0ef1bc386 2026-11-09)`, **`host: x86_64-pc-windows-gnu`** (confirmed verbatim via `rustc -vV` and `rustup show`: only `stable-x86_64-pc-windows-gnu` and `1.89.0-x86_64-pc-windows-gnu` are installed on this host). GitHub Actions `ci.yml` adds `ubuntu-latest`, `macos-latest` (`aarch64-apple-darwin`), and `windows-latest` (`x86_64-pc-windows-msvc`) — currently `PENDING_REMOTE_CI`. |
| **`CROSS_COMPILE_VERIFIED`** | **`PENDING_REMOTE_CI`** | `.github/workflows/embedded.yml` targets `thumbv7em-none-eabihf`, `riscv32imac-unknown-none-elf`, `aarch64-unknown-none`, and `wasm32-unknown-unknown`. Awaits first push to `SM-Acoustics/rtconv-simd`. |
| **`DEVICE_VERIFIED`** | **`NO` (`OPEN`)** | No physical MCU/SBC/ESP32-P4 hardware execution has been performed. |

## 2. Release Identification & Provenance

| Field | Value |
| :--- | :--- |
| **Project / Package / Lib** | `RTConv SIMD` / `rtconv-simd` / `rtconv_simd` (`0.1.0`, `MIT OR Apache-2.0`) |
| **Intended GitHub Repository** | `SM-Acoustics/rtconv-simd` *(not confirmed created; `git ls-remote` returns `Repository not found` and `gh` CLI is not installed)* |
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
| **G1** | Name availability on `crates.io` & GitHub repo | **BLOCKED** | `crates.io`: `rtconv-simd` and `rtconv_simd` available (HTTP 404). `SM-Acoustics/rtconv-simd` not yet verified created (`gh` not installed; `git ls-remote` exit 128); `repository` and `documentation` held commented in `Cargo.toml` until confirmed. |
| **G2** | Pure `#![no_std]`, zero `alloc`, `#![forbid(unsafe_code)]`, `default = []`, MSRV `1.89.0` | **PASS** | `cargo +1.89.0 check --lib --no-default-features` (`EXIT 0`), `cargo +1.89.0 check --all-features` (`EXIT 0`), `cargo check --lib --no-default-features` (`EXIT 0`), `cargo check --all-features` (`EXIT 0`). |
| **G3** | Formatting & Clippy zero-warnings | **PASS** | `cargo fmt --all -- --check`, `cargo clippy --lib --no-default-features -- -D warnings`, `cargo clippy --all-targets --all-features -- -D warnings`: all `EXIT 0`. |
| **G4** | Full numerical, streaming, boundary & state-transition suite | **PASS** | Stable `1.100.0` (`--release`): `--no-default-features` **21/21 ok**, `--all-features` **22/22 ok**. MSRV `1.89.0`: `--no-default-features` **21/21 ok**, `--all-features` **22/22 ok**. *(Test count reflects consolidating `tests/no_alloc_rt.rs` into one sequential entry point covering all 3 backend scenarios without TLS recursion on Rust 1.89.0.)* |
| **G5** | Zero heap allocations in `prepare` & callbacks | **PASS** | `cargo test --release --all-features --test no_alloc_rt -- --test-threads=1` (`EXIT 0`) and under default parallel execution on both `1.89.0` and `1.100.0`. Verifies 0 allocs / 0 deallocs across `MicroFftBackend`, `SimdFixedFftBackend` (`SPREAD`), and `PhastFftBackend` callbacks. |
| **G6** | Runnable public examples | **PASS** | `direct_fir`, `hybrid_convolution`, `backend_selection --all-features`, `multiple_instances`: all `EXIT 0`. |
| **G7** | Public API docs without warnings | **PASS** | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`: `EXIT 0`. |
| **G8** | Performance & SIMD evidence without overclaiming | **PASS** | WO-002 benchmarks & disassembly (`FMA=96`, `YMM=527`, `XMM=5251`) documented in `docs/BENCHMARKS.md` and `docs/SIMD_BACKENDS.md` under `HOST_TESTED` (`x86_64-pc-windows-gnu`, Ryzen 3 PRO 2300U). Disclaims `HARD_RT_CERTIFIED` and RISC-V RVV SIMD. |
| **G9** | License & supply-chain audit (`SM-ENG-005` Proposed v1) | **PASS** | `LICENSE-MIT` + `LICENSE-APACHE` packaged; 100% permissive licenses (`MIT`, `Apache-2.0`, `Unicode-3.0`), zero C/C++/FFI, `phastft` with `default-features = false` (no `rayon`). |
| **G10** | Clean `.crate` packaging & dry-run (no `--allow-dirty`) | **PASS** | `cargo package --list`, `cargo package`, and `cargo publish --dry-run`: **45 files** packaged cleanly (`EXIT 0`). Expected Cargo metadata warning remains until G1 (`repository`/`documentation`) is uncommented. |
| **G11** | Independent consumer crate validation | **PASS** | Fresh consumer crate depending on unpacked `target/package/rtconv-simd-0.1.0` compiled and ran `Direct`, `Partitioned` (`SimdFixedFftBackend`), and `Hybrid` (`SPREAD`) in `--release`: `CONSUMER_OK ... simd_active=true` (`EXIT 0`). |
| **G12** | Remote CI (`ci.yml`, `embedded.yml`, `release-check.yml`) | **`PENDING_REMOTE_CI`** | `ci.yml` covers `ubuntu-latest`, `macos-latest`, `windows-latest`, and dedicated MSRV `1.89.0` job; `embedded.yml` covers 4 `no_std` cross-targets; `release-check.yml` includes `msrv-check` + dry-run packaging. Awaits push to `SM-Acoustics/rtconv-simd`. |
| **G13** | Owner approval, GitHub publication & `crates.io` publish | **`AWAITING_OWNER_APPROVAL`** | No `git push`, no merge, no public visibility change, and no real `cargo publish` executed. |
