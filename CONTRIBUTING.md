# Contributing to `rtconv-simd`

Thank you for your interest in contributing to `rtconv-simd`. This crate is maintained under the SM-Acoustics organization as an independent L1 DSP library (aligned with the proposed organization standards in [`SM-Acoustics/.github`](https://github.com/SM-Acoustics/.github), currently *Proposed v1*).

## 1. Rust Setup

- Install stable Rust **1.89 or newer** (`rustup toolchain install stable`) with `rustfmt` and `clippy`.
- Optional embedded targets for compile checks:
  ```bash
  rustup target add thumbv7em-none-eabihf riscv32imac-unknown-none-elf aarch64-unknown-none wasm32-unknown-unknown
  ```

## 2. Running Tests

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo check --lib --no-default-features
cargo test --release --no-default-features
cargo test --release --all-features
cargo test --release --all-features --test no_alloc_rt -- --test-threads=1
```

## 3. Running Benchmarks

```bash
cargo bench --bench rtconv_bench --all-features
cargo bench --bench fft_backends_bench --all-features
```

Always report CPU/MCU model, OS/RTOS, `rustc --version`, target triple, profile/flags, and whether evidence is `HOST_ONLY` or `DEVICE_VERIFIED`.

## 4. Reproducing Numerical Issues

Include:
1. Exact impulse response (or deterministic seed/generator) and input length.
2. Engine, block size, `FftBackend`, `SimdBackend`, and `tail_spreading` setting.
3. Measured `max_abs`, RMS, and SNR against the independent `f64` reference.

## 5. Proposing SIMD Optimizations & Hardware Evidence

- Preserve `#![no_std]` and `#![forbid(unsafe_code)]` in the core library.
- Benchmark **before and after** across both SSE2 and AVX2 (or target ISA) and inspect generated assembly (`cargo rustc --lib --release -- --emit=asm`).
- Do not promote an optimization to default if it regresses latency, memory, or p99 predictability on representative targets.
- Never claim SIMD support for an architecture (such as RISC-V RVV) without real implementation and target evidence.

## 6. Pull Requests & ADR Criteria

- Work on a scoped branch (`fix/...`, `feat/...`, `docs/...`) targeting `main`.
- Fill out the PR checklist with reproducible commands and provenance labels (`PROVEN`, `INFERRED`, `OPEN`).
- An Architecture Decision Record (ADR) is required for breaking public API changes, changes to workspace ownership/layout semantics, new mandatory core dependencies, or any proposal to relax `#![forbid(unsafe_code)]` or `#![no_std]`.
