#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <mpi.h>
#include <omp.h>

typedef struct {
    float val;
    int pos;
} local_max;

/* THIS FUNCTION CAN BE MODIFIED */
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int first_k = (rank * layer_size) / size;
    int final_k = ((rank + 1) * layer_size) / size;
    int local_size = final_k - first_k;

    float *layer = (float *)malloc((local_size + 2) * sizeof(float));
    float *layer_copy = (float *)malloc((local_size + 2) * sizeof(float)); 
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    #pragma omp parallel for schedule(static)
    for( int k=0; k<local_size+2; k++ ) {
        layer[k] = 0.0f;
        layer_copy[k] = 0.0f;
    }

    int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;
    int left  = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    MPI_Request req[4];
    float threshold = THRESHOLD / (float)layer_size; 

    for( int i=0; i<num_storms; i++) {

        local_max local_values;
        local_values.val = 0.0f;
        local_values.pos = -1;
        
        int storm_len = storms[i].size;
        float *storm_energies = (float*)malloc(storm_len * sizeof(float));
        int *storm_positions  = (int*)malloc(storm_len * sizeof(int));
        
        for(int j=0; j<storm_len; j++) {
            storm_positions[j] = storms[i].posval[j*2];
            
            float raw_energy = (float)storms[i].posval[j*2+1] * 1000.0f;
            storm_energies[j] = raw_energy / layer_size;
        }

        #pragma omp parallel
        {   
            #pragma omp for schedule(static)
            for( int k=first_k; k<final_k; k++ ) {
                int local_k = k - first_k + 1;

                for( int j=0; j<storm_len; j++ ) {
                    int distance = abs(storm_positions[j] - k) + 1;
                    float atenuacion = sqrtf((float)distance);
                    float energy_k = storm_energies[j] / atenuacion; 

                    if ( energy_k >= threshold || energy_k <= -threshold ) {
                        layer[local_k] += energy_k;
                    }
                }
                layer_copy[local_k] = layer[local_k];
            }

            #pragma omp master
            {
            MPI_Isend(&layer_copy[1], 1, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &req[0]);
            MPI_Isend(&layer_copy[local_size], 1, MPI_FLOAT, right, 1, MPI_COMM_WORLD, &req[1]);
            MPI_Irecv(&layer_copy[0], 1, MPI_FLOAT, left, 1, MPI_COMM_WORLD, &req[2]);
            MPI_Irecv(&layer_copy[local_size+1], 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &req[3]);
            }

            #pragma omp for schedule(static)
            for (int k = 2; k < local_size; k++) {
                layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
            }

            #pragma omp barrier
            #pragma omp master
            {
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            }

            #pragma omp single
            {
                int k = 1;
                if (first_k + k - 1 == 0) 
                    layer[k] = layer_copy[k];
                else 
                    layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;

                k = local_size;
                if (first_k + k - 1 == layer_size - 1) 
                    layer[k] = layer_copy[k];
                else 
                    layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
            }

            float thread_val = 0.0f;
            int thread_pos = 0;

            #pragma omp for schedule(static)
            for (int k=1; k<=local_size; k++) {
                int global_k = first_k + k - 1;
                if (global_k > 0 && global_k < layer_size - 1) {
                    if (layer[k] > layer[k-1] && layer[k] > layer[k+1]) {
                        if (layer[k] > thread_val) {
                            thread_val = layer[k];
                            thread_pos = global_k;
                        }
                    }
                }
            }
            
            #pragma omp critical
            {
                if (thread_val > local_values.val) {
                    local_values.val = thread_val;
                    local_values.pos = thread_pos;
                }
            }
        }

        local_max final_values;
        MPI_Allreduce(&local_values, &final_values, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        maximum[i]   = final_values.val;
        positions[i] = final_values.pos;

        free( storm_energies );
        free( storm_positions );
    }

    free( layer );
    free( layer_copy );
}
