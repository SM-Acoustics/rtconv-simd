#!/usr/bin/env bash
# FPU-less binary guard (issue #32): compile-only, no hardware access.
#
# Fails when the given ELF (the hardware/esp32c6 harness, riscv32imac) links
#   - any symbol of the f32 auto path that can never execute there
#     (TW_COS_Q, SimdFixedFftBackend, POWER_OF_FIVE_128),
#   - any RISC-V soft-float routine (__mulsf3, __muldf3 and the rest of the
#     libgcc/compiler-builtins float ABI: add/sub/mul/div, comparisons, conversions),
#   - or more than BUDGET bytes of .text + .rodata (default 81,196 B = rtconv-simd 0.2.0;
#     `none` reports the sizes without a budget, e.g. for the opt-in benchmark build).
# Writes the section sizes (llvm-size -A) to $GITHUB_STEP_SUMMARY when set.
#
# Usage: scripts/check-no-fpu-binary.sh <elf> [budget_bytes]
# Tools: $LLVM_NM / $LLVM_SIZE (default: llvm-nm / llvm-size on PATH); $GUARD_LABEL names the
# build variant in the job summary.
set -euo pipefail

elf="${1:?usage: $0 <elf> [budget_bytes]}"
budget="${2:-81196}"
nm="${LLVM_NM:-llvm-nm}"
size="${LLVM_SIZE:-llvm-size}"

symbols="$("$nm" -C "$elf")"
fail=0

for sym in TW_COS_Q SimdFixedFftBackend POWER_OF_FIVE_128 __mulsf3 __muldf3; do
  if hits="$(grep -F -- "$sym" <<<"$symbols")"; then
    echo "::error::forbidden symbol '$sym' linked into $elf:"
    echo "$hits"
    fail=1
  fi
done

softfloat_re=' (__(add|sub|mul|div|neg|pow)[sd]f[23]|__(eq|ne|lt|le|gt|ge|unord|cmp)[sd]f2|__float(un)?(si|di|ti)[sd]f|__fix(uns)?[sd]f(si|di|ti)|__extendsfdf2|__truncdfsf2)$'
if hits="$(grep -E -- "$softfloat_re" <<<"$symbols")"; then
  echo "::error::soft-float routines linked into $elf:"
  echo "$hits"
  fail=1
fi

sizes="$("$size" -A "$elf")"
section() { awk -v s="$1" '$1 == s { print $2; found = 1 } END { if (!found) print 0 }' <<<"$sizes"; }
text="$(section .text)"
rodata="$(section .rodata)"
bss="$(section .bss)"
total="$(section Total)"
code=$((text + rodata))

echo "$sizes"
echo ".text + .rodata = $code B (budget $budget)"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### ESP32-C6 harness${GUARD_LABEL:+ ($GUARD_LABEL)} size (\`llvm-size -A\`, riscv32imac, compile-only)"
    echo
    echo "| Section | Bytes |"
    echo "| :--- | ---: |"
    echo "| \`.text\` | $text |"
    echo "| \`.rodata\` | $rodata |"
    echo "| \`.bss\` | $bss |"
    echo "| \`.text + .rodata\` | $code (budget $budget) |"
    echo "| total (\`size -A\`) | $total |"
    echo
    if [[ $fail -eq 0 ]]; then
      echo "Forbidden-symbol guard: none of \`TW_COS_Q\`, \`SimdFixedFftBackend\`, \`POWER_OF_FIVE_128\`, \`__mulsf3\`, \`__muldf3\` or any other soft-float routine is linked."
    else
      echo "Forbidden-symbol guard: **FAILED** (see job log)."
    fi
  } >>"$GITHUB_STEP_SUMMARY"
fi

if [[ "$budget" != "none" ]] && ((code > budget)); then
  echo "::error::.text + .rodata = $code B exceeds the $budget B budget"
  fail=1
fi
exit "$fail"
