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

## Agentic software-assurance checklist
- [ ] AI assistance and exact changed modules disclosed
- [ ] Requirement/source, affected production entry point and reachability recorded
- [ ] New stubs/TODO/dead code audited; unsupported work fails explicitly
- [ ] Independent reference/oracle plus negative/edge/regression evidence provided
- [ ] No tests, gate, CI or security checks silently weakened
- [ ] Unsafe/FFI, DMA/MMIO, concurrency, allocations, deadlines and supply chain reviewed where applicable
- [ ] Capability status (UNSUPPORTED / IMPLEMENTED_UNVERIFIED / VERIFIED / FAILED) and exact HOST/QEMU/DEVICE scope stated
- [ ] Commands, targets, versions, failure receipts, hashes and residual risks provided
- [ ] No hardware action, merge, deployment or release without human approval
