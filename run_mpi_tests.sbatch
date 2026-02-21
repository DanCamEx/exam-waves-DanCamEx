#!/usr/bin/env bash

set -euo pipefail

NODES="${NODES:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  salloc --nodes="$NODES" --cpus-per-task="$CPUS_PER_TASK" --time="$TIME_LIMIT" --qos=students_limit \
    bash "$0" "$@"
  exit $?
fi
export CPATH="${CPATH:-}"
export MANPATH="${MANPATH:-}"
source /home/guest/init-hpc.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

RESULTS_LOG="results_mpi_omp_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"

# Associative array to store sequential times per test group
declare -A SEQ_TIMES

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

set_fixed_sequential_times() {
  SEQ_TIMES["${TEST_GROUPS[0]}"]="4.2788356000"
  SEQ_TIMES["${TEST_GROUPS[1]}"]="35.6124945000"
  SEQ_TIMES["${TEST_GROUPS[2]}"]="4.7702027000"
}

run_mpi_omp() {
  local test_group="$1"
  local num_procs="$2"
  local num_threads="$3"
  local -a wave_files=()
  read -r -a wave_files <<< "$test_group"
  local layer_size
  layer_size="$(get_layer_size "$test_group")"
  local sum_time=0
  local avg_time=0
  local seq_time
  seq_time="${SEQ_TIMES[$test_group]:-3.40}"
  local total_cores
  total_cores=$((num_procs * num_threads))

  echo "[MPI+OpenMP] files=${test_group} size=${layer_size} procs=${num_procs} threads=${num_threads}" | tee -a "$RESULTS_LOG"
  for i in $(seq 1 10); do
    echo "[MPI+OpenMP] run=${i}" | tee -a "$RESULTS_LOG"
    OMP_NUM_THREADS="$num_threads" run_output=$(mpirun -np "$num_procs" --map-by node:PE="$num_threads" --bind-to core \
      ./energy_storms_mpi_omp "$layer_size" "${wave_files[@]}")
    echo "$run_output" | tee -a "$RESULTS_LOG"
    run_time=$(echo "$run_output" | awk '/^Time:/{print $2; exit}')
    if [[ -n "${run_time}" ]]; then
      sum_time=$(awk -v s="$sum_time" -v t="$run_time" 'BEGIN {printf "%.10f", s + t}')
    else
      echo "[MPI+OpenMP] Warning: Time not found for run=${i}" | tee -a "$RESULTS_LOG"
    fi
  done

  avg_time=$(awk -v s="$sum_time" 'BEGIN {printf "%.10f", s / 10.0}')
  local speedup efficiency
  speedup=$(awk -v seq="$seq_time" -v par="$avg_time" 'BEGIN {printf "%.4f", seq / par}')
  efficiency=$(awk -v speedup="$speedup" -v cores="$total_cores" 'BEGIN {printf "%.4f", speedup / cores}')
  
  echo "[MPI+OpenMP] average_time=${avg_time}" | tee -a "$RESULTS_LOG"
  echo "[MPI+OpenMP] speedup=${speedup}x (vs seq_time=${seq_time}s)" | tee -a "$RESULTS_LOG"
  echo "[MPI+OpenMP] efficiency=${efficiency} (${total_cores} cores)" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
}

echo "Building MPI+OpenMP binary..." | tee -a "$RESULTS_LOG"
make energy_storms_mpi_omp | tee -a "$RESULTS_LOG"

set_fixed_sequential_times
echo "Using fixed sequential baselines: 4.2788356000, 35.6124945000, 4.7702027000" | tee -a "$RESULTS_LOG"

echo "Starting MPI+OpenMP tests..." | tee -a "$RESULTS_LOG"
for test_group in "${TEST_GROUPS[@]}"; do
  echo "=== Testing ${test_group} ===" | tee -a "$RESULTS_LOG"
  
  # Test with 2 processes
  for num_threads in 2 4 8 16 32; do
    run_mpi_omp "$test_group" 2 "$num_threads"
  done
  
  # Test with 4 processes
  for num_threads in 2 4 8 16 32; do
    run_mpi_omp "$test_group" 4 "$num_threads"
  done
done

echo "All tests completed." | tee -a "$RESULTS_LOG"
