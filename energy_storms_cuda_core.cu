#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

/* Funzione atomica per calcolare il massimo tra due float */
__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/* Kernel che divide le tempeste in blocchi e calcola l'impatto per ogni particella su ogni cella */
__global__ void add_impact(float *d_layer, float *d_layer_copy, Storm *d_storms, int layer_size, int i) {

    __shared__ int s_pos[BLOCK_SIZE];
    __shared__ float s_energy[BLOCK_SIZE];

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    float threshold = THRESHOLD / layer_size;

    for (int t = 0; t < d_storms[i].size; t += BLOCK_SIZE) {
        int particles_per_block = min(BLOCK_SIZE, d_storms[i].size - t);

        for (int idx = threadIdx.x; idx < particles_per_block; idx += blockDim.x) {
            int global_idx = t + idx;
            int pos = d_storms[i].posval[global_idx * 2];
            float energy = (float)d_storms[i].posval[global_idx * 2 + 1] * 1000.0f / layer_size;
            s_pos[idx] = pos;
            s_energy[idx] = energy;
        }
        __syncthreads();

        if (k < layer_size) {
            float local_energy = d_layer[k];
            for (int j = 0; j < particles_per_block; j++) {
                int distance = s_pos[j] - k;
                if (distance < 0) distance = -distance;
                distance = distance + 1;

                float atenuacion = sqrtf((float)distance);
                float energy_k = s_energy[j] / atenuacion;

                if (fabsf(energy_k) >= threshold) {
                    local_energy += energy_k;
                }
            }
            d_layer[k] = local_energy;
        }
        __syncthreads();
    }

    if (k < layer_size) {
        d_layer_copy[k] = d_layer[k];
    }
}

/* Kernel per aggiornare i l'energia delle celle facendone la media con i vicini adiacenti */
__global__ void update_layer(float *d_layer, float *d_layer_copy, int layer_size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k > 0 && k < layer_size - 1 ) {
        d_layer[k] = ( d_layer_copy[k-1] + d_layer_copy[k] + d_layer_copy[k+1] ) / 3;
    }
}

/* Kernel per trovare il massimo valore nel layer */
__global__ void find_max_val(float *d_layer, int layer_size, float *d_maximum, int i) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k > 0 && k < layer_size - 1 ) {
        if ( d_layer[k] > d_layer[k-1] && d_layer[k] > d_layer[k+1] ) {
            if ( d_layer[k] > d_maximum[i] ) {
                atomicMax(&d_maximum[i], d_layer[k]);
            }
        }
    }
}

/* Kernel per trovare la posizione del massimo valore */
__global__ void find_max_pos(float *d_layer, int layer_size, int *d_positions, float max_found, int i) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k > 0 && k < layer_size - 1 ) {
        if ( d_layer[k] > d_layer[k-1] && d_layer[k] > d_layer[k+1] ) {
            if ( d_layer[k] == max_found ) {
                atomicExch(&d_positions[i], k);
            }
        }
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    float *d_layer;
    float *d_layer_copy;
    cudaMalloc(&d_layer, sizeof(float) * layer_size);
    cudaMalloc(&d_layer_copy, sizeof(float) * layer_size);
    cudaMemset(d_layer, 0, sizeof(float) * layer_size);
    cudaMemset(d_layer_copy, 0, sizeof(float) * layer_size);
    
    Storm *d_storms;
    cudaMalloc(&d_storms, sizeof(Storm) * num_storms);
    cudaMemcpy(d_storms, storms, sizeof(Storm) * num_storms, cudaMemcpyHostToDevice);
    
    
    float *d_maximum;
    cudaMalloc(&d_maximum, sizeof(float) * num_storms);
    cudaMemcpy(d_maximum, maximum, sizeof(float) * num_storms, cudaMemcpyHostToDevice);
    
    int *d_positions;
    cudaMalloc(&d_positions, sizeof(int) * num_storms);
    cudaMemcpy(d_positions, positions, sizeof(int) * num_storms, cudaMemcpyHostToDevice);
    

    Storm *temp_storms = (Storm *)malloc(sizeof(Storm) * num_storms);
    memcpy(temp_storms, storms, sizeof(Storm) * num_storms);
    
    for( int i=0; i<num_storms; i++) {
        int *d_posval;
        cudaMalloc(&d_posval, storms[i].size * 2 * sizeof(int));
        cudaMemcpy(d_posval, storms[i].posval, storms[i].size * 2 * sizeof(int), cudaMemcpyHostToDevice);
        temp_storms[i].posval = d_posval; 
    }
    cudaMemcpy(d_storms, temp_storms, sizeof(Storm) * num_storms, cudaMemcpyHostToDevice);

    for( int i=0; i<num_storms; i++) {

        add_impact<<<(layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_layer, d_layer_copy, d_storms, layer_size, i);
        cudaDeviceSynchronize();
        
        update_layer<<<(layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_layer, d_layer_copy, layer_size);
        cudaDeviceSynchronize();
        
        find_max_val<<<(layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_layer, layer_size, d_maximum, i);
        cudaDeviceSynchronize();
        
        float max_found;
        cudaMemcpy(&max_found, &d_maximum[i], sizeof(float), cudaMemcpyDeviceToHost);

        find_max_pos<<<(layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_layer, layer_size, d_positions, max_found, i);
        cudaDeviceSynchronize();
       
    }

    cudaMemcpy(maximum, d_maximum, sizeof(float) * num_storms, cudaMemcpyDeviceToHost);
    cudaMemcpy(positions, d_positions, sizeof(int) * num_storms, cudaMemcpyDeviceToHost);
}
