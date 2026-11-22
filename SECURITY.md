# Security Policy

This policy aligns with the SM-Acoustics organization security guidelines ([`SM-Acoustics/.github/SECURITY.md`](https://github.com/SM-Acoustics/.github/blob/main/SECURITY.md)).

## Reporting a Vulnerability

If the repository offers GitHub **Private Vulnerability Reporting**, use its Security tab. Do not publish exploits, credentials, vulnerability details, or personal data in public issues.

If private vulnerability reporting is unavailable, contact a project maintainer through an independently verified private channel before public disclosure. This document does not claim that a particular email inbox or vulnerability-reporting feature is enabled.

## Scope & Threat Model

`rtconv-simd` is a pure-Rust `#![no_std]`, `#![forbid(unsafe_code)]` DSP library with no network, filesystem, or FFI surface. Relevant security and safety considerations include:

- Supply-chain integrity of dependencies (`fearless_simd`, `fearless_simd_macros`, `microfft`, `libm`, and optional `phastft`).
- Panic-free and arithmetic-overflow-safe handling of untrusted impulse responses (e.g., `NaN`/`Inf` rejection during `prepare()`, checked memory calculations).
- Protection against unbounded execution time or unexpected heap allocations in real-time audio callbacks.
