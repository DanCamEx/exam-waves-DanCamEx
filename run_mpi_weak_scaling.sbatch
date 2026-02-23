#!/usr/bin/env bash

set -euo pipefail

NODES="${NODES:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
REPEAT="${REPEAT:-5}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  salloc --nodes="$NODES" --cpus-per-task="$CPUS_PER_TASK" --time="$TIME_LIMIT" --qos=students_limit \
    bash "$0" "$@"
  exit $?
fi

export CPATH="${CPATH:-}"
export MANPATH="${MANPATH:-}"
source /home/guest/init-hpc.sh

export OMP_PLACES=cores
export OMP_PROC_BIND=close

RESULTS_LOG="results_mpi_omp_weak_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
OPT_FLAGS="-O3 -march=native -fno-math-errno -fno-trapping-math -fno-signaling-nans -funroll-loops"

SEQ_TIMES_GROUP_1=(4.25 8.54 17.05 34.14 68.25 136.33)
SEQ_TIMES_GROUP_2=(35.50 71.26 142.50 285 570 1140)

# Weak scaling test groups (100M excluded)
TEST_GROUPS=(
  "test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5"
  "test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4 test_files/test_07_a1M_p5k_w1"
)

get_base_layer_size() {
  case "$1" in
    *a30k*) echo 30000 ;;
    *a1M*) echo 1000000 ;;
    *) echo 20000 ;;
  esac
}

# Families of combinations required:
# - fixed 2 MPI ranks, up to 32 threads per process
# - fixed 4 MPI ranks, up to 32 threads per process
# (mpi_ranks x omp_threads)
CORE_CONFIGS_2P=(
  "2 1"
  "2 2"
  "2 4"
  "2 8"
  "2 16"
  "2 32"
)

CORE_CONFIGS_4P=(
  "4 1"
  "4 2"
  "4 4"
  "4 8"
  "4 16"
  "4 32"
)

get_fixed_seq_time() {
  local group_index="$1"
  local step_index="$2"

  if [[ "$group_index" -eq 0 ]]; then
    echo "${SEQ_TIMES_GROUP_1[$step_index]}"
  else
    echo "${SEQ_TIMES_GROUP_2[$step_index]}"
  fi
}

run_weak_case() {
  local test_group="$1"
  local mpi_ranks="$2"
  local omp_threads="$3"
  local layer_size="$4"
  local group_index="$5"
  local step_index="$6"

  local -a wave_files=()
  read -r -a wave_files <<< "$test_group"

  local sum_time=0
  local avg_time
  local total_cores=$((mpi_ranks * omp_threads))
  local seq_time
  local speedup
  local efficiency
  local mpirun_status

  seq_time="$(get_fixed_seq_time "$group_index" "$step_index")"

  echo "[MPI+OpenMP][WEAK] files=${test_group}" | tee -a "$RESULTS_LOG"
  echo "[MPI+OpenMP][WEAK] layer=${layer_size} ranks=${mpi_ranks} threads=${omp_threads} total_cores=${total_cores}" | tee -a "$RESULTS_LOG"

  for i in $(seq 1 "$REPEAT"); do
    echo "[MPI+OpenMP][WEAK] run=${i}" | tee -a "$RESULTS_LOG"
    set +e
    OMP_NUM_THREADS="$omp_threads" run_output=$(mpirun -np "$mpi_ranks" --map-by node:PE="$omp_threads" --bind-to core \
      ./energy_storms_mpi_omp "$layer_size" "${wave_files[@]}" 2>&1)
    mpirun_status=$?
    set -e

    if [[ "$mpirun_status" -ne 0 ]]; then
      echo "[MPI+OpenMP][WEAK] Error: mpirun failed (exit=${mpirun_status}) for layer=${layer_size}, ranks=${mpi_ranks}, threads=${omp_threads}, run=${i}" | tee -a "$RESULTS_LOG"
      echo "$run_output" | tee -a "$RESULTS_LOG"
      return 1
    fi

    echo "$run_output" | tee -a "$RESULTS_LOG"

    run_time=$(echo "$run_output" | awk '/^Time:/{print $2; exit}')
    if [[ -n "${run_time}" ]]; then
      sum_time=$(awk -v s="$sum_time" -v t="$run_time" 'BEGIN {printf "%.10f", s + t}')
    else
      echo "[MPI+OpenMP][WEAK] Warning: Time not found for run=${i}" | tee -a "$RESULTS_LOG"
    fi
  done

  avg_time=$(awk -v s="$sum_time" -v n="$REPEAT" 'BEGIN {printf "%.10f", s / n}')
  speedup=$(awk -v seq="$seq_time" -v par="$avg_time" 'BEGIN {printf "%.4f", seq / par}')
  efficiency=$(awk -v sp="$speedup" -v c="$total_cores" 'BEGIN {printf "%.4f", sp / c}')

  echo "[MPI+OpenMP][WEAK] seq_time=${seq_time}" | tee -a "$RESULTS_LOG"
  echo "[MPI+OpenMP][WEAK] average_time=${avg_time}" | tee -a "$RESULTS_LOG"
  echo "[MPI+OpenMP][WEAK] speedup=${speedup}x" | tee -a "$RESULTS_LOG"
  echo "[MPI+OpenMP][WEAK] efficiency=${efficiency}" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
}

echo "Compiling MPI+OpenMP binary (no make)..." | tee -a "$RESULTS_LOG"
mpicc $OPT_FLAGS -fopenmp energy_storms_mpi_omp.c energy_storms_mpi_omp_core.c -lm -o energy_storms_mpi_omp | tee -a "$RESULTS_LOG"
echo "Using provided fixed sequential baselines per layer step." | tee -a "$RESULTS_LOG"

echo "Starting MPI+OpenMP weak-scaling tests (particles fixed, layer and resources doubled)..." | tee -a "$RESULTS_LOG"

for group_index in "${!TEST_GROUPS[@]}"; do
  test_group="${TEST_GROUPS[$group_index]}"
  echo "=== Weak scaling group: ${test_group} ===" | tee -a "$RESULTS_LOG"

  base_layer_size="$(get_base_layer_size "$test_group")"

  echo "--- Fixed MPI ranks: 2 ---" | tee -a "$RESULTS_LOG"
  layer_size="$base_layer_size"
  step_index=0
  for config in "${CORE_CONFIGS_2P[@]}"; do
    read -r mpi_ranks omp_threads <<< "$config"
    if ! run_weak_case "$test_group" "$mpi_ranks" "$omp_threads" "$layer_size" "$group_index" "$step_index"; then
      echo "[MPI+OpenMP][WEAK] Skipping failed configuration and continuing..." | tee -a "$RESULTS_LOG"
    fi
    step_index=$((step_index + 1))
    layer_size=$((layer_size * 2))
  done

  echo "--- Fixed MPI ranks: 4 ---" | tee -a "$RESULTS_LOG"
  layer_size="$base_layer_size"
  step_index=0
  for config in "${CORE_CONFIGS_4P[@]}"; do
    read -r mpi_ranks omp_threads <<< "$config"
    if ! run_weak_case "$test_group" "$mpi_ranks" "$omp_threads" "$layer_size" "$group_index" "$step_index"; then
      echo "[MPI+OpenMP][WEAK] Skipping failed configuration and continuing..." | tee -a "$RESULTS_LOG"
    fi
    step_index=$((step_index + 1))
    layer_size=$((layer_size * 2))
  done
done

echo "All weak-scaling tests completed." | tee -a "$RESULTS_LOG"
