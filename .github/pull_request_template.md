## Summary
- Problem solved and scope of changes.

## Real-Time & `#![no_std]` Checklist
- [ ] Core crate compiles with `cargo check --lib --no-default-features` (zero `std`, zero `alloc`).
- [ ] `#![forbid(unsafe_code)]` preserved in `src/lib.rs`.
- [ ] Zero heap allocations/deallocations verified (`cargo test --release --test no_alloc_rt -- --test-threads=1`).
- [ ] No blocking locks, threads, I/O, or panics introduced on valid audio buffers.
- [ ] Numerical tests against `f64` reference pass (`cargo test --release --all-features`).

## Evidence
- Commit / Toolchain / Target / Hardware:
- Commands executed and summary of results:
- Provenance label (`PROVEN` / `INFERRED` / `OPEN`; `HOST_ONLY` / `DEVICE_VERIFIED`):
