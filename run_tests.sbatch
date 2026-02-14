#!/usr/bin/env bash
#SBATCH --job-name=energy_storms_tests
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

source /home/guest/init-hpc.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

RESULTS_LOG="results_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"

TEST_FILES=(
  "test_files/test_02_a30k_p20k_w1"
  "test_files/test_07_a1M_p5k_w1"
  "test_files/test_08_a100M_p1_w1"
  "test_files/test_01_a35_p5_w3"
)

get_layer_size() {
  case "$1" in
    *a35*) echo 35 ;;
    *a30k*) echo 30000 ;;
    *a1M*) echo 1000000 ;;
    *a100M*) echo 100000000 ;;
    *) echo 20000 ;;
  esac
}

run_mpi_omp() {
  local test_file="$1"
  local layer_size
  layer_size="$(get_layer_size "$test_file")"
  local sum_time=0
  local avg_time=0

  echo "[MPI+OpenMP] file=${test_file} size=${layer_size}" | tee -a "$RESULTS_LOG"
  for i in $(seq 1 10); do
    echo "[MPI+OpenMP] run=${i}" | tee -a "$RESULTS_LOG"
    run_output=$(mpirun -np 4 --map-by node:PE=32 --bind-to core \
      ./energy_storms_mpi_omp "$layer_size" "$test_file")
    echo "$run_output" | tee -a "$RESULTS_LOG"
    run_time=$(echo "$run_output" | awk '/^Time:/{print $2; exit}')
    if [[ -n "${run_time}" ]]; then
      sum_time=$(awk -v s="$sum_time" -v t="$run_time" 'BEGIN {printf "%.10f", s + t}')
    else
      echo "[MPI+OpenMP] Warning: Time not found for run=${i}" | tee -a "$RESULTS_LOG"
    fi
  done

  avg_time=$(awk -v s="$sum_time" 'BEGIN {printf "%.10f", s / 10.0}')
  echo "[MPI+OpenMP] average_time=${avg_time}" | tee -a "$RESULTS_LOG"
}

run_cuda() {
  local test_file="$1"
  local layer_size
  layer_size="$(get_layer_size "$test_file")"
  local sum_time=0
  local avg_time=0

  echo "[CUDA] file=${test_file} size=${layer_size}" | tee -a "$RESULTS_LOG"
  for i in $(seq 1 10); do
    echo "[CUDA] run=${i}" | tee -a "$RESULTS_LOG"
    run_output=$(srun --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=32 \
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
}

echo "Building binaries..." | tee -a "$RESULTS_LOG"
make energy_storms_mpi_omp energy_storms_cuda | tee -a "$RESULTS_LOG"

echo "Starting MPI+OpenMP tests..." | tee -a "$RESULTS_LOG"
for test_file in "${TEST_FILES[@]}"; do
  run_mpi_omp "$test_file"
done

echo "Starting CUDA tests..." | tee -a "$RESULTS_LOG"
for test_file in "${TEST_FILES[@]}"; do
  run_cuda "$test_file"
done

echo "All tests completed." | tee -a "$RESULTS_LOG"
