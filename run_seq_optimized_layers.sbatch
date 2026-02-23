#!/usr/bin/env bash

set -euo pipefail

REPEAT="${REPEAT:-5}"
RESULTS_LOG="results_seq_opt_layers_local_$(date +%Y%m%d_%H%M%S).log"
OPT_FLAGS="-O3 -march=native -fno-math-errno -fno-trapping-math -fno-signaling-nans -funroll-loops"

TEST_GROUPS=(
  "test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5"
  "test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4 test_files/test_07_a1M_p5k_w1"
)

LAYER_MULTIPLIERS=(1 2 4 8 16 32)

get_base_layer_size() {
  case "$1" in
    *a30k*) echo 30000 ;;
    *a1M*) echo 1000000 ;;
    *) echo 20000 ;;
  esac
}

run_seq_opt_case() {
  local test_group="$1"
  local layer_size="$2"

  local -a wave_files=()
  local sum_time=0
  local avg_time

  read -r -a wave_files <<< "$test_group"

  echo "[SEQ-OPT] files=${test_group}" | tee -a "$RESULTS_LOG"
  echo "[SEQ-OPT] layer=${layer_size}" | tee -a "$RESULTS_LOG"

  for i in $(seq 1 "$REPEAT"); do
    echo "[SEQ-OPT] run=${i}" | tee -a "$RESULTS_LOG"
    run_output=$(./energy_storms_optimized "$layer_size" "${wave_files[@]}")
    echo "$run_output" | tee -a "$RESULTS_LOG"

    run_time=$(echo "$run_output" | awk '/^Time:/{print $2; exit}')
    if [[ -n "${run_time}" ]]; then
      sum_time=$(awk -v s="$sum_time" -v t="$run_time" 'BEGIN {printf "%.10f", s + t}')
    else
      echo "[SEQ-OPT] Warning: Time not found for run=${i}" | tee -a "$RESULTS_LOG"
    fi
  done

  avg_time=$(awk -v s="$sum_time" -v n="$REPEAT" 'BEGIN {printf "%.10f", s / n}')
  echo "[SEQ-OPT] average_time=${avg_time}" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
}

echo "Compiling optimized sequential binary (no make)..." | tee -a "$RESULTS_LOG"
gcc $OPT_FLAGS energy_storms_optimized.c energy_storms_core_optimized.c -lm -o energy_storms_optimized | tee -a "$RESULTS_LOG"

echo "Starting optimized sequential layer-scaling tests..." | tee -a "$RESULTS_LOG"
for test_group in "${TEST_GROUPS[@]}"; do
  base_layer_size="$(get_base_layer_size "$test_group")"
  echo "=== Seq optimized group: ${test_group} ===" | tee -a "$RESULTS_LOG"

  for mult in "${LAYER_MULTIPLIERS[@]}"; do
    layer_size=$((base_layer_size * mult))
    run_seq_opt_case "$test_group" "$layer_size"
  done
done

echo "All optimized sequential layer-scaling tests completed." | tee -a "$RESULTS_LOG"
