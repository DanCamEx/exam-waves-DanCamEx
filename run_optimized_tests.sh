#!/usr/bin/env bash

set -euo pipefail

export MANPATH="${MANPATH:-}"
export CPATH="${CPATH:-}"
source /home/guest/init-hpc.sh

RESULTS_LOG="results_optimized_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"

TEST_GROUPS=(
  "test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5"
  "test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4 test_files/test_07_a1M_p5k_w1"
  "test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2 test_files/test_08_a100M_p1_w3 test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2"
)

get_layer_size() {
  case "$1" in
    *a30k*) echo 30000 ;;
    *a1M*) echo 1000000 ;;
    *a100M*) echo 100000000 ;;
    *) echo 20000 ;;
  esac
}

build_optimized_if_needed() {
  local bin="./energy_storms_optimized"
  local src_main="energy_storms_optimized.c"
  local src_core="energy_storms_core_optimized.c"
  local hdr="energy_storms.h"

  if [[ ! -x "$bin" || "$src_main" -nt "$bin" || "$src_core" -nt "$bin" || "$hdr" -nt "$bin" ]]; then
    echo "Building optimized sequential binary (gcc) ..." | tee -a "$RESULTS_LOG"
    gcc -O3 -march=native -fno-math-errno -fno-trapping-math -fno-signaling-nans -funroll-loops "$src_main" "$src_core" -lm -o "$bin"
    echo "Optimized sequential binary built." | tee -a "$RESULTS_LOG"
  else
    echo "Optimized sequential binary already up-to-date." | tee -a "$RESULTS_LOG"
  fi
}

run_optimized() {
  local test_group="$1"
  local -a wave_files=()
  read -r -a wave_files <<< "$test_group"
  local layer_size
  layer_size="$(get_layer_size "$test_group")"
  local sum_time=0
  local avg_time=0

  echo "[OPTIMIZED] files=${test_group} size=${layer_size}" | tee -a "$RESULTS_LOG"
  for i in $(seq 1 10); do
    echo "[OPTIMIZED] run=${i}" | tee -a "$RESULTS_LOG"
    run_output=$(./energy_storms_optimized "$layer_size" "${wave_files[@]}")
    echo "$run_output" | tee -a "$RESULTS_LOG"
    run_time=$(echo "$run_output" | awk '/^Time:/{print $2; exit}')
    if [[ -n "${run_time}" ]]; then
      sum_time=$(awk -v s="$sum_time" -v t="$run_time" 'BEGIN {printf "%.10f", s + t}')
    else
      echo "[OPTIMIZED] Warning: Time not found for run=${i}" | tee -a "$RESULTS_LOG"
    fi
  done

  avg_time=$(awk -v s="$sum_time" 'BEGIN {printf "%.10f", s / 10.0}')
  echo "[OPTIMIZED] average_time=${avg_time}" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
}

build_optimized_if_needed

echo "Starting optimized sequential tests..." | tee -a "$RESULTS_LOG"
for test_group in "${TEST_GROUPS[@]}"; do
  run_optimized "$test_group"
done

echo "All optimized sequential tests completed." | tee -a "$RESULTS_LOG"
