#!/usr/bin/env bash

set -euo pipefail

export MANPATH="${MANPATH:-}"
export CPATH="${CPATH:-}"
source /home/guest/init-hpc.sh

RESULTS_LOG="results_cuda_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"

TEST_FILES=(
  "test_files/test_02_a30k_p20k_w1"
  "test_files/test_07_a1M_p5k_w1"
  "test_files/test_08_a100M_p1_w1"
)

# Associative array to store sequential times per test file
declare -A SEQ_TIMES

get_layer_size() {
  case "$1" in
    *a30k*) echo 30000 ;;
    *a1M*) echo 1000000 ;;
    *a100M*) echo 100000000 ;;
    *) echo 20000 ;;
  esac
}

run_sequential() {
  local test_file="$1"
  local layer_size
  layer_size="$(get_layer_size "$test_file")"
  local sum_time=0
  local avg_time=0

  echo "[SEQUENTIAL] file=${test_file} size=${layer_size}" | tee -a "$RESULTS_LOG"
  for i in $(seq 1 10); do
    echo "[SEQUENTIAL] run=${i}" | tee -a "$RESULTS_LOG"
    run_output=$(./energy_storms_optimized "$layer_size" "$test_file")
    echo "$run_output" | tee -a "$RESULTS_LOG"
    run_time=$(echo "$run_output" | awk '/^Time:/{print $2; exit}')
    if [[ -n "${run_time}" ]]; then
      sum_time=$(awk -v s="$sum_time" -v t="$run_time" 'BEGIN {printf "%.10f", s + t}')
    else
      echo "[SEQUENTIAL] Warning: Time not found for run=${i}" | tee -a "$RESULTS_LOG"
    fi
  done

  avg_time=$(awk -v s="$sum_time" 'BEGIN {printf "%.10f", s / 10.0}')
  SEQ_TIMES["$test_file"]="$avg_time"

  echo "[SEQUENTIAL] average_time=${avg_time}" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
}

run_cuda() {
  local test_file="$1"
  local layer_size
  layer_size="$(get_layer_size "$test_file")"
  local sum_time=0
  local avg_time=0
  local seq_time
  seq_time="${SEQ_TIMES[$test_file]:-3.40}"
  local speedup efficiency
  local total_cores=1

  echo "[CUDA] file=${test_file} size=${layer_size}" | tee -a "$RESULTS_LOG"
  for i in $(seq 1 10); do
    echo "[CUDA] run=${i}" | tee -a "$RESULTS_LOG"
    run_output=$(srun --qos=students_limit --gpus=1 \
      ./energy_storms_cuda "$layer_size" "$test_file")
    echo "$run_output" | tee -a "$RESULTS_LOG"
    run_time=$(echo "$run_output" | awk '/^Time:/{print $2; exit}')
    if [[ -n "${run_time}" ]]; then
      sum_time=$(awk -v s="$sum_time" -v t="$run_time" 'BEGIN {printf "%.10f", s + t}')
    else
      echo "[CUDA] Warning: Time not found for run=${i}" | tee -a "$RESULTS_LOG"
    fi
  done

  avg_time=$(awk -v s="$sum_time" 'BEGIN {printf "%.10f", s / 10.0}')
  echo "[CUDA] average_time=${avg_time}" | tee -a "$RESULTS_LOG"
  speedup=$(awk -v seq="$seq_time" -v par="$avg_time" 'BEGIN {printf "%.4f", seq / par}')
  efficiency=$(awk -v speedup="$speedup" -v cores="$total_cores" 'BEGIN {printf "%.4f", speedup / cores}')
  echo "[CUDA] speedup=${speedup}x (vs seq_time=${seq_time}s)" | tee -a "$RESULTS_LOG"
  echo "[CUDA] efficiency=${efficiency} (${total_cores} device)" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
}

echo "Building CUDA binary..." | tee -a "$RESULTS_LOG"
make energy_storms_cuda | tee -a "$RESULTS_LOG"

echo "Building Sequential binary..." | tee -a "$RESULTS_LOG"
make energy_storms_optimized | tee -a "$RESULTS_LOG"

echo "Starting Sequential baseline tests..." | tee -a "$RESULTS_LOG"
for test_file in "${TEST_FILES[@]}"; do
  run_sequential "$test_file"
done

echo "Starting CUDA tests..." | tee -a "$RESULTS_LOG"
for test_file in "${TEST_FILES[@]}"; do
  run_cuda "$test_file"
done

echo "All CUDA tests completed." | tee -a "$RESULTS_LOG"
