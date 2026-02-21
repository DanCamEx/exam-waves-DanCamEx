#!/usr/bin/env bash

set -euo pipefail

export MANPATH="${MANPATH:-}"
export CPATH="${CPATH:-}"
source /home/guest/init-hpc.sh

RESULTS_LOG="results_cuda_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"

TEST_GROUPS=(
  "test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5"
  "test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4 test_files/test_07_a1M_p5k_w1"
  "test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2 test_files/test_08_a100M_p1_w3 test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2"
)

# Associative array to store sequential times per test group
declare -A SEQ_TIMES

get_layer_size() {
  case "$1" in
    *a30k*) echo 30000 ;;
    *a1M*) echo 1000000 ;;
    *a100M*) echo 100000000 ;;
    *) echo 20000 ;;
  esac
}

set_fixed_sequential_times() {
  SEQ_TIMES["${TEST_GROUPS[0]}"]="4.2788356000"
  SEQ_TIMES["${TEST_GROUPS[1]}"]="35.6124945000"
  SEQ_TIMES["${TEST_GROUPS[2]}"]="4.7702027000"
}

run_cuda() {
  local test_group="$1"
  local -a wave_files=()
  read -r -a wave_files <<< "$test_group"
  local layer_size
  layer_size="$(get_layer_size "$test_group")"
  local sum_time=0
  local avg_time=0
  local seq_time
  seq_time="${SEQ_TIMES[$test_group]:-3.40}"
  local speedup efficiency
  local total_cores=1

  echo "[CUDA] files=${test_group} size=${layer_size}" | tee -a "$RESULTS_LOG"
  for i in $(seq 1 10); do
    echo "[CUDA] run=${i}" | tee -a "$RESULTS_LOG"
    run_output=$(srun --qos=students_limit --gpus=1 \
      ./energy_storms_cuda "$layer_size" "${wave_files[@]}")
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

set_fixed_sequential_times
echo "Using fixed sequential baselines: 4.2788356000, 35.6124945000, 4.7702027000" | tee -a "$RESULTS_LOG"

echo "Building CUDA binary..." | tee -a "$RESULTS_LOG"
make energy_storms_cuda | tee -a "$RESULTS_LOG"

echo "Starting CUDA tests..." | tee -a "$RESULTS_LOG"
for test_group in "${TEST_GROUPS[@]}"; do
  run_cuda "$test_group"
done

echo "All CUDA tests completed." | tee -a "$RESULTS_LOG"
