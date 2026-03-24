#!/usr/bin/env bash
# Build a markdown table: matrix dimensions + packed Answer benchmark times.
# Requires: CGO, CUDA, gpu/build/libsimplepir_cutlass.a, Go with -tags cutlass.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CGO_ENABLED=1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

GO="${GO:-go}"
D="${D:-256}"
LOG_NS="${LOG_NS:-16 17 18 19 20 21 22}"

echo "## Packed SimplePIR Answer: matrix sizes and times (D=${D})"
echo ""
echo "Rows/columns are **uint32 elements** (after DB squish, factor 3)."
echo ""
echo "**D_squished** is the left-hand matrix in \`MatrixMulVecPacked(D, Q)\` (server DB slice)."
echo "**Q** is a column vector of length \`D_squished_cols × 3\` (padded query; matches SimplePIR \`Query\`)."
echo ""
echo "Times are per call on this machine (run \`eval/gen_answer_matrix_table.sh\` to regenerate)."
echo ""
echo "| LOG_N | L×M (raw DB) | **D_squished (L×C)** | Q length | CPU ns/op | GPU+H2D(D) ns/op | GPU resident D ns/op | CUDA kernel_ms | CUDA e2e_ms |"
echo "|------:|---------------|----------------------|---------:|----------:|-----------------:|---------------------:|---------------:|------------:|"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

for n in $LOG_NS; do
  row="$("$GO" run ./eval/dimensions "$n" 2>&1 | grep -E '^[[:digit:]]' | tail -1)"
  if [[ -z "$row" ]]; then
    continue
  fi
  read -r logn d l m ds_r ds_c qel <<<$(echo "$row" | tr '\t' ' ')

  out="$(LOG_N="$n" D="$D" "$GO" test -tags cutlass -count=1 -run '^$' \
    -bench BenchmarkSimplePIRPackedAnswerOnlineOnly -benchtime 10x ./pir 2>/dev/null)" || true
  cpu="$(echo "$out" | awk '/CPU_packed_D_host/{getline; if(/ns\/op/)print $2}')"
  gpuup="$(echo "$out" | awk '/GPU_packed_upload_D_each_iter/{getline; if(/ns\/op/)print $2}')"
  gpures="$(echo "$out" | awk '/GPU_packed_resident_D/{getline; if(/ns\/op/)print $2}')"

  ev="$(LOG_N="$n" D="$D" "$GO" test -tags cutlass -count=1 -run '^$' -bench BenchmarkPackedAnswerResidentCUDAEvents -benchtime 12x ./pir 2>/dev/null)" || true
  evline="$(echo "$ev" | grep 'ns/op' | tail -1)"
  kem="$(echo "$evline" | sed -n 's/.* \([0-9.]*\) e2e_ms.*/\1/p')"
  kkm="$(echo "$evline" | sed -n 's/.* \([0-9.]*\) kernel_ms.*/\1/p')"

  printf '| %s | %s×%s | **%s×%s** | %s | %s | %s | %s | %s | %s |\n' \
    "$logn" "$l" "$m" "$ds_r" "$ds_c" "$qel" "${cpu:-—}" "${gpuup:-—}" "${gpures:-—}" "${kkm:-—}" "${kem:-—}"

done | tee "$TMP"

echo ""
echo "_CUDA columns: \`BenchmarkPackedAnswerResidentCUDAEvents\` (device kernel_ms = after Q H2D until kernel end; e2e_ms = full copy+sync)._"
