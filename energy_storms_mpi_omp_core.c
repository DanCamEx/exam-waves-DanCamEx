#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <mpi.h>
#include <omp.h>

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k, rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int k_start = (rank * layer_size) / size;
    int k_end = ((rank + 1) * layer_size) / size;
    int local_size = k_end - k_start;
    
    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * local_size );
    float *layer_copy = (float *)malloc( sizeof(float) * local_size ); 
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    #pragma omp parallel for schedule(static) 
    for( k=0; k<local_size; k++ ) {
        layer[k] = 0.0f;
        layer_copy[k] = 0.0f;
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
    
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {
        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        #pragma omp parallel for reduction(+:layer[:local_size]) private(k) schedule (guided)
        for( j=0; j<storms[i].size; j++ ) {
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            int position = storms[i].posval[j*2]; 

            /* For each cell */
            for( k=0; k<local_size; k++ ) {
                int distance = position - (k_start + k);
                if ( distance < 0 ) distance = - distance;
                distance = distance + 1;
                float atenuacion = sqrtf( (float)distance );
                float energy_k = energy / layer_size / atenuacion;
                if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
                    layer[k] += energy_k;
            }
        }
        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        float *temp = layer_copy;
        layer_copy = layer;
        layer = temp;

        /* 4.2.2. Do a Sendrecv to send borders values to neighbors,
                and receive them at the same time */
        float border_left = 0.0f;
        float border_right = 0.0f;
        MPI_Request req[4];

        MPI_Isend(&layer_copy[0], 1, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&layer_copy[local_size-1], 1, MPI_FLOAT, right, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(&border_right, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(&border_left, 1, MPI_FLOAT, left, 1, MPI_COMM_WORLD, &req[3]);

        #pragma omp parallel for schedule(static)
        for( k=1; k<local_size-1; k++ )
            layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3;

        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        /* 4.2.4. Update layer using the ancillary values.
                Updates the first and last positions of the current local layer, 
                except the left for rank 0 and the right for the last rank */
        if (rank > 0 && local_size > 1)
            layer[0] = (border_left + layer_copy[0] + layer_copy[1]) / 3;
        if (rank < size - 1 && local_size > 1)
            layer[local_size-1] = (layer_copy[local_size-2] + layer_copy[local_size-1] + border_right) / 3;
        
        /* 4.3. Locate the maximum value in the layer, and its position */
        float local_max = 0.0f;
        int local_pos = 0;

        #pragma omp parallel
        {
            float thread_max = 0.0f;
            int thread_pos = 0;
            
            #pragma omp for nowait
            for ( k=0; k<local_size; k++ ) {
                if (layer[k] > thread_max) { 
                    thread_max = layer[k]; 
                    thread_pos = k + k_start; 
                }
            }
            
            #pragma omp critical
            if (thread_max > local_max) {
                local_max = thread_max;
                local_pos = thread_pos;
            }
        }

        struct { float val; int pos; } in = { local_max, local_pos }, out;
        MPI_Allreduce(&in, &out, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        maximum[i]   = out.val;
        positions[i] = out.pos;
}

    /* 5. Free allocated memory */
    free(layer);
    free(layer_copy);
}
