# --- Execution Parameters ---
export OMP_NUM_THREADS=32
MPI_PROCS=4

# --- MPI Run Flags ---
MPIRUN_FLAGS =  -np $(MPI_PROCS) \
                --map-by ppr:1:node:PE=32 


# --- Compiler Flags ---
# Flags for MPI+OpenMP code
# Uncomment and add extra flags if you need them
MPI_OMP_EXTRA_CFLAGS = -O3
#MPI_OMP_EXTRA_LIBS =

# Flags for CUDA code
# Uncomment and add extra flags if you need them
CUDA_EXTRA_CFLAGS = -O3
#CUDA_EXTRA_LIBS =