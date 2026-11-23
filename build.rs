//! Classifies the compilation target as having a hardware FPU or not.
//!
//! Emits `cfg(rtconv_no_fpu)` for targets whose `f32` arithmetic would be software-emulated, so
//! `rtconv_simd::auto` can pick the Q1.15 / block-floating-point engines there.
//!
//! On stable Rust the RISC-V `F`/`D` extensions are not exposed as `cfg(target_feature)`
//! (`riscv32imafc-unknown-none-elf`, used by ESP32-P4, and `riscv32imac-unknown-none-elf`, used
//! by ESP32-C6, report identical cfgs), so the ISA string of the target triple is inspected.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rustc-check-cfg=cfg(rtconv_no_fpu)");

    let target = env::var("TARGET").unwrap_or_default();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let abi = env::var("CARGO_CFG_TARGET_ABI").unwrap_or_default();
    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has_feature = |name: &str| features.split(',').any(|f| f == name);
    let force_q15 = env::var_os("CARGO_FEATURE_NATIVE_Q15").is_some();

    let has_fpu = match arch.as_str() {
        "riscv32" | "riscv64" => {
            // ISA letters follow the width, e.g. "riscv32imafc" -> "imafc", "riscv64gc" -> "gc".
            let isa = target
                .split('-')
                .next()
                .unwrap_or("")
                .trim_start_matches("riscv32")
                .trim_start_matches("riscv64");
            has_feature("f")
                || has_feature("d")
                || isa
                    .split('_')
                    .next()
                    .unwrap_or("")
                    .contains(['f', 'd', 'g'])
        }
        "arm" => {
            abi == "eabihf"
                || ["vfp2", "vfp3", "vfp4", "fp-armv8"]
                    .iter()
                    .any(|f| has_feature(f))
        }
        "avr" | "msp430" => false,
        // ESP32-S2 (Xtensa LX7) has no FPU; ESP32 and ESP32-S3 do.
        "xtensa" => !target.contains("esp32s2"),
        _ => true,
    };

    if force_q15 || !has_fpu {
        println!("cargo:rustc-cfg=rtconv_no_fpu");
    }
}
