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

/* 4.1. Aggiungi le energie degli impatti alle celle del layer
    Carica le particelle in shared memory a blocchi e accumula
    il contributo energetico su ogni cella */
__global__ void add_impact(float *d_layer, float *d_layer_copy, Storm *d_storms, int layer_size, int i) {

    __shared__ int s_pos[BLOCK_SIZE];
    __shared__ float s_energy[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float threshold = THRESHOLD / layer_size;

    /* Scorriamo le particelle della tempesta i a blocchi di BLOCK_SIZE */
    for (int t = 0; t < d_storms[i].size; t += BLOCK_SIZE) {
        int particles_per_block = min(BLOCK_SIZE, d_storms[i].size - t);

        /* Copiamo in shared memory posizione ed energia delle particelle correnti */
        for (int idx = threadIdx.x; idx < particles_per_block; idx += blockDim.x) {
            int global_idx = t + idx;
            int pos = d_storms[i].posval[global_idx * 2];
            float energy = (float)d_storms[i].posval[global_idx * 2 + 1] * 1000.0f / layer_size;
            s_pos[idx] = pos;
            s_energy[idx] = energy;
        }
        __syncthreads();

        /* Aggiorna la propria cella del layer */
        for (int k = tid; k < layer_size; k += stride) {
            float local_energy = d_layer[k];
            for (int j = 0; j < particles_per_block; j++) {
                /* 1. Calcola la distanza */
                int distance = s_pos[j] - k;
                if (distance < 0) distance = -distance;
                distance = distance + 1;

                /* 2. Calcola l'energia attenuata */
                float atenuacion = sqrtf((float)distance);
                float energy_k = s_energy[j] / atenuacion;

                /* 3. Non aggiungere se il valore assoluto è inferiore alla soglia */
                if (fabsf(energy_k) >= threshold) {
                    local_energy += energy_k;
                }
            }
            /* Scrivi l'energia accumulata della cella k */
            d_layer[k] = local_energy;
        }
        __syncthreads();
    }

    /* Copiamo il layer aggiornato nell'array per la fase di rilassamento */
    for (int k = tid; k < layer_size; k += stride) {
        d_layer_copy[k] = d_layer[k];
    }
}

/* 4.2. Rilassamento dell'energia tra le tempeste
    4.2.3. Aggiorna il layer facendo la media con i vicini adiacenti */
__global__ void update_layer(float *d_layer, float *d_layer_copy, int layer_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int k = tid; k < layer_size; k += stride) {
        if ( k > 0 && k < layer_size - 1 ) {
            d_layer[k] = ( d_layer_copy[k-1] + d_layer_copy[k] + d_layer_copy[k+1] ) / 3;
        }
    }
}

/* 4.3. Trova il valore massimo nel layer
    Controlla solo i massimi locali e aggiorna il massimo globale della tempesta i */
__global__ void find_max_val(float *d_layer, int layer_size, float *d_maximum, int i) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int k = tid; k < layer_size; k += stride) {
        if ( k > 0 && k < layer_size - 1 ) {
            if ( d_layer[k] > d_layer[k-1] && d_layer[k] > d_layer[k+1] ) {
                if ( d_layer[k] > d_maximum[i] ) {
                    atomicMax(&d_maximum[i], d_layer[k]);
                }
            }
        }
    }
}

/* 4.4. Trova la posizione associata al massimo della tempesta i */
__global__ void find_max_pos(float *d_layer, int layer_size, int *d_positions, float *d_maximum, int i) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int k = tid; k < layer_size; k += stride) {
        if ( k > 0 && k < layer_size - 1 ) {
            if ( d_layer[k] > d_layer[k-1] && d_layer[k] > d_layer[k+1] ) {
                if ( d_layer[k] == d_maximum[i] ) {
                    atomicExch(&d_positions[i], k);
                }
            }
        }
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    /* Decidi la dimensione della griglia in base alla GPU corrente */
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    int blocks_per_sm = device_prop.maxThreadsPerMultiProcessor / BLOCK_SIZE;

    const int WAVE_BLOCKS = device_prop.multiProcessorCount * blocks_per_sm;
    const int TARGET_GRID_BLOCKS = WAVE_BLOCKS * 2;

    int min_grid_blocks = (layer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int target_blocks;
    if (min_grid_blocks > TARGET_GRID_BLOCKS) {
        target_blocks = min_grid_blocks;
    } else {
        target_blocks = TARGET_GRID_BLOCKS;
    }
    int grid_blocks = ((target_blocks + WAVE_BLOCKS - 1) / WAVE_BLOCKS) * WAVE_BLOCKS;

    /* Alloca la memoria principale sul device e inizializza i layer a zero */
    float *d_layer;
    float *d_layer_copy;
    cudaMalloc(&d_layer, sizeof(float) * layer_size);
    cudaMalloc(&d_layer_copy, sizeof(float) * layer_size);
    cudaMemset(d_layer, 0, sizeof(float) * layer_size);
    cudaMemset(d_layer_copy, 0, sizeof(float) * layer_size);
    
    /* Copia la struttura delle tempeste dall'host al device */
    Storm *d_storms;
    cudaMalloc(&d_storms, sizeof(Storm) * num_storms);
    cudaMemcpy(d_storms, storms, sizeof(Storm) * num_storms, cudaMemcpyHostToDevice);
    
    
    /* Alloca i vettori dei massimi e relative posizioni sul device */
    float *d_maximum;
    cudaMalloc(&d_maximum, sizeof(float) * num_storms);
    cudaMemcpy(d_maximum, maximum, sizeof(float) * num_storms, cudaMemcpyHostToDevice);
    
    int *d_positions;
    cudaMalloc(&d_positions, sizeof(int) * num_storms);
    cudaMemcpy(d_positions, positions, sizeof(int) * num_storms, cudaMemcpyHostToDevice);
    
    /* Crea una copia delle tempeste per sostituire i puntatori posval
        con i relativi puntatori in memoria device */
    Storm *temp_storms = (Storm *)malloc(sizeof(Storm) * num_storms);
    memcpy(temp_storms, storms, sizeof(Storm) * num_storms);
    
    for( int i=0; i<num_storms; i++) {
        int *d_posval;
        cudaMalloc(&d_posval, storms[i].size * 2 * sizeof(int));
        cudaMemcpy(d_posval, storms[i].posval, storms[i].size * 2 * sizeof(int), cudaMemcpyHostToDevice);
        temp_storms[i].posval = d_posval; 
    }
    cudaMemcpy(d_storms, temp_storms, sizeof(Storm) * num_storms, cudaMemcpyHostToDevice);

    /* 4. Inizio della simulazione delle tempeste */
    for( int i=0; i<num_storms; i++) {

        /* 4.1. Aggiungi le energie degli impatti alle celle del layer */
        add_impact<<<grid_blocks, BLOCK_SIZE>>>(d_layer, d_layer_copy, d_storms, layer_size, i);
        
        /* 4.2. Rilassamento dell'energia tra le tempeste */
        update_layer<<<grid_blocks, BLOCK_SIZE>>>(d_layer, d_layer_copy, layer_size);
        
        /* 4.3. Trova il valore massimo nel layer */
        find_max_val<<<grid_blocks, BLOCK_SIZE>>>(d_layer, layer_size, d_maximum, i);

        /* 4.4. Trova la posizione del massimo */
        find_max_pos<<<grid_blocks, BLOCK_SIZE>>>(d_layer, layer_size, d_positions, d_maximum, i);
       
    }

    /* Copia i risultati finali dal device all'host */
    cudaMemcpy(maximum, d_maximum, sizeof(float) * num_storms, cudaMemcpyDeviceToHost);
    cudaMemcpy(positions, d_positions, sizeof(int) * num_storms, cudaMemcpyDeviceToHost);

    /* Libera la memoria */
    for (int i = 0; i < num_storms; i++) {
        cudaFree(temp_storms[i].posval);
    }
    free(temp_storms);
    cudaFree(d_positions);
    cudaFree(d_maximum);
    cudaFree(d_storms);
    cudaFree(d_layer_copy);
    cudaFree(d_layer);
}

