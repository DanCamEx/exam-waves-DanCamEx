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
/* Function to update a single position of the layer */
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int first_k = (rank * layer_size) / size;
    int final_k = ((rank + 1) * layer_size) / size;
    int local_size = final_k - first_k;

    float *layer = (float *)calloc(local_size + 2, sizeof(float));
    float *layer_copy = (float *)calloc(local_size + 2, sizeof(float)); 
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int right, left = MPI_PROC_NULL;
    if (rank > 0) {
        left = rank - 1;
    } else {
        left = MPI_PROC_NULL;
    }
    if (rank < size - 1) {
        right = rank + 1;
    } else {
        right = MPI_PROC_NULL;
    }
    
    for( int i=0; i<num_storms; i++) {

        #pragma omp parallel for schedule(dynamic)
        for( int j=0; j<storms[i].size; j++ ) {
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            int position = storms[i].posval[j*2];

            for( int k=first_k; k<final_k; k++ ) {
                int distance = position - k;
                if ( distance < 0 ) distance = - distance;
                distance = distance + 1;
              
                float atenuacion = sqrtf( (float)distance );
                float energy_k = energy / layer_size / atenuacion;

                if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
                    #pragma omp atomic
                    layer[k - first_k + 1] += energy_k;
            } 
        }
        
        #pragma omp parallel for
        for (int k = 1; k <= local_size; k++) 
            layer_copy[k] = layer[k];

        MPI_Request req[4];
        MPI_Isend(&layer_copy[1], 1, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&layer_copy[local_size], 1, MPI_FLOAT, right, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(&layer_copy[0], 1, MPI_FLOAT, left, 1, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(&layer_copy[local_size+1], 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        
        #pragma omp parallel for
        for (int k = 1; k <= local_size; k++) {
            int global_k = first_k + k - 1;
            if (global_k == 0 || global_k == layer_size - 1) {
                layer[k] = layer_copy[k];
            } else {
                layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3;
            }
        }

        MPI_Isend(&layer[1], 1, MPI_FLOAT, left, 2, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&layer[local_size], 1, MPI_FLOAT, right, 3, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(&layer[0], 1, MPI_FLOAT, left, 3, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(&layer[local_size+1], 1, MPI_FLOAT, right, 2, MPI_COMM_WORLD, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
       
        local_max local_values;
        local_values.val = 0.0f;
        local_values.pos = 0;
        
        #pragma omp parallel
        {
            float thread_val = 0.0f;
            int thread_pos = 0;

            #pragma omp for
            for (int k = 1; k <= local_size; k++) {
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
    }

    free( layer );
    free( layer_copy );
}
