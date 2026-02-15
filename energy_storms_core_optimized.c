#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions, float* layer, float* layer_copy) {
    int i, j, k;
    float threshold = THRESHOLD / layer_size;

    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {

        /* Allochiamo lo spazio per calcolare le energie e le posizioni dei vari impatti*/
        float *energies = (float*)malloc(sizeof(float) * storms[i].size);
        int *impact_positions = (int*)malloc(sizeof(int) * storms[i].size);

		/* Calcoliamo una sola volta le energie e le posizioni degli impatti */
        for(j=0; j<storms[i].size; j++){
            float energy = (float)storms[i].posval[j*2+1] * 1000.0f;
            energies[j] = energy / layer_size;
            impact_positions[j] = storms[i].posval[j*2];
        }

		/* 4.1. Aggiungi le energie degli impatti alle celle del layer */
		/* Per ogni particella */
		for( k=1; k<=layer_size; k++ ) {
			/* Inizializziamo l'energia locale della cella */
			float local_energy = layer[k];

			/* Per ogni cella nel layer */
			for( j=0; j<storms[i].size; j++ ) {

				 /* 1. Calcola la distanza (intero) */
                int distance = abs(impact_positions[j] - k) + 1;

                /* 3. Calcolo effettivo solo se necessario */
                float atenuacion = sqrtf( (float)distance );
                float energy_k = energies[j] / atenuacion;

				/* 5. Non aggiungere se il valore assoluto è inferiore alla soglia */
				if ( fabs(energy_k) >= threshold ) 
					local_energy += energy_k;
			}
			/* Solo alla fine scriviamo sull'array l'energia totale accumulata nella cella k */
			layer[k] = local_energy;
		}
		free(energies);
        free(impact_positions);

        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        for( k=0; k<layer_size; k++ ) 
            layer_copy[k] = layer[k];

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        for( k=1; k<layer_size-1; k++ )
            layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;

        /* 4.3. Locate the maximum value in the layer, and its position */
        for( k=1; k<layer_size-1; k++ ) {
            /* Check it only if it is a local maximum */
            if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
                if ( layer[k] > maximum[i] ) {
                    maximum[i] = layer[k];
                    positions[i] = k;
                }
            }
        }
    }
}