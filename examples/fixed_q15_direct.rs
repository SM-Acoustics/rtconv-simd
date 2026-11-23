#[cfg(feature = "fixed-q15")]
fn main() {
    use rtconv_simd::fixed::{f32_to_q15, q15_to_f32};
    use rtconv_simd::{FixedQ15DirectConfig, FixedQ15DirectConvolver};

    let ir = [
        f32_to_q15(0.5),
        f32_to_q15(-0.25),
        f32_to_q15(0.125),
        f32_to_q15(0.0625),
    ];
    let cfg = FixedQ15DirectConfig { max_ir_len: 32 };
    let req = FixedQ15DirectConvolver::memory_requirements(cfg).unwrap();
    let mut ws = [0i16; 64];

    let mut conv = FixedQ15DirectConvolver::prepare(&ir, cfg, &mut ws[..req.total_i16]).unwrap();
    let mut input = [0i16; 32];
    input[0] = f32_to_q15(0.5);
    let mut output = [0i16; 32];
    conv.process_block(&input, &mut output).unwrap();

    println!(
        "Q15 Direct: taps={}, ws={} B, y[0..4]=[{:.4}, {:.4}, {:.4}, {:.4}], sat={}",
        ir.len(),
        req.total_bytes,
        q15_to_f32(output[0]),
        q15_to_f32(output[1]),
        q15_to_f32(output[2]),
        q15_to_f32(output[3]),
        conv.saturation_count()
    );
}

#[cfg(not(feature = "fixed-q15"))]
fn main() {
    println!("Run with `--features fixed-q15` to execute the Q15 Direct FIR example.");
}
