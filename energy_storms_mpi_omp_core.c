#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "energy_storms.h"

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
	int i, j, k, rank, size;
	float threshold = THRESHOLD / layer_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* 3. Assegna una porzione del layer ad ogni processo MPI e
		ricalcola la dimensione locale */
	int first_k = (rank * layer_size) / size;
	int last_k = ((rank + 1) * layer_size) / size;
	int local_size = last_k - first_k;

	/* Alloca la memoria e la inizializza a 0, aggiungendo due "ghost cells" usate per lo scambio 
		dei bordi più avanti */
	float *layer = (float *)malloc(sizeof(float) * (local_size + 2));
	float *layer_copy = (float *)malloc(sizeof(float) * (local_size + 2));
	if (layer == NULL || layer_copy == NULL) {
		fprintf(stderr, "Error: Allocating the layer memory\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	#pragma omp parallel for schedule(static)
	for(k=0; k<local_size+2; k++) {
		layer[k] = 0.0f;
		layer_copy[k] = 0.0f;
	}

	/* Decide i processi di destra e di sinistra.
		Per il primo processo e l'ultimo viene assegnato un rank speciale che MPI
		usa per indicare l'assenza del processo */
	int left_rank, right_rank;
	if (rank > 0)
		left_rank = rank - 1;
	else
		left_rank = MPI_PROC_NULL;
	if (rank < size - 1)
		right_rank = rank + 1;
	else
		right_rank = MPI_PROC_NULL;


	/* 4. Inizio della simulazione delle tempeste */
	for(i=0; i<num_storms; i++) {
		/* Allochiamo lo spazio per calcolare le energie e le posizioni dei vari impatti*/
        float *energies = (float*)malloc(sizeof(float) * storms[i].size);
        int *impact_positions = (int*)malloc(sizeof(int) * storms[i].size);

		/* Calcoliamo una sola volta le energie e le posizioni degli impatti */
        for(j=0; j<storms[i].size; j++){
            float energy = (float)storms[i].posval[j*2+1] * 1000.0f;
            energies[j] = energy / layer_size;
            impact_positions[j] = storms[i].posval[j*2] - first_k + 1;
        }

		/* 4.1. Aggiungi le energie degli impatti alle celle del layer */
		/* Per ogni particella */
		#pragma omp parallel for private(j) schedule(static)
		for(k=1; k<=local_size; k++) {
			/* Inizializziamo l'energia locale della cella */
			float local_energy = layer[k];

			/* Per ogni cella nel layer */
			for(j=0; j<storms[i].size; j++) {

				 /* 1. Calcola la distanza (intero) */
                int distance = abs(impact_positions[j] - k) + 1;

                /* 2. Calcola l'energia attenuata */
                float atenuacion = sqrtf((float)distance);
                float energy_k = energies[j] / atenuacion;

				/* 3. Non aggiungere se il valore assoluto è inferiore alla soglia */
				if (fabs(energy_k) >= threshold) 
					local_energy += energy_k;
			}
			/* Scrivi sull'array l'energia totale accumulata nella cella k */
			layer[k] = local_energy;
		}
		free(energies);
        free(impact_positions);
		

		/* 4.2. Rilassamento dell'energia tra le tempeste */
		/* 4.2.1. Copia i valori nell'array copia */
		#pragma omp parallel for schedule(static) private(k)
		for(k=1; k<=local_size; k++) 
			layer_copy[k] = layer[k];

		/* Scambia verso SINISTRA: Tutti inviano al vicino di sinistra e ricevono dal vicino di destra contemporaneamente */
		MPI_Sendrecv(&layer_copy[1], 1, MPI_FLOAT, left_rank, 1,
					&layer_copy[local_size+1], 1, MPI_FLOAT, right_rank, 1,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Scambia verso DESTRA: Tutti inviano al vicino di destra e ricevono dal vicino di sinistra contemporaneamente */
		MPI_Sendrecv(&layer_copy[local_size], 1, MPI_FLOAT, right_rank, 2,
					&layer_copy[0], 1, MPI_FLOAT, left_rank, 2,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* 4.2.3. Aggiorna il layer usando i valori della copia. Vengono aggiornati i valori ai bordi
				usando le "ghost cells" allocate all'inizio */
		#pragma omp parallel for schedule(static) private(k)
		for(k=1; k<=local_size; k++) {
			layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;
		}

		/* 4.3. Trova il valore massimo nel layer */
		/* Crea una struct per avere il tipo di dato corretto nella MPI_Allreduce */
		struct {float val; int pos;} local_max, global_max;
		local_max.val = 0;
		local_max.pos = 0;

		#pragma omp parallel private(k)
		{
			float thread_max_val = 0;
			int thread_max_pos = 0;

			/* Trova il massimo valore nel layer locale */
			#pragma omp for schedule(static) nowait
			for(k=1; k<=local_size; k++) {
				/* Controlla solo se è un massimo locale */
				if (layer[k] > layer[k-1] && layer[k] > layer[k+1]) {
					if (layer[k] > thread_max_val) {
						thread_max_val = layer[k];
						thread_max_pos = first_k + k - 1;
					}
				}
			}
			
			/* Aggiorna il valore locale solo se il valore massimo
				trovato dal thread è maggiore di quello locale */
			#pragma omp critical
			{
				if ( thread_max_val > local_max.val ) {
					local_max.val = thread_max_val;
					local_max.pos = thread_max_pos;
				}
			}
		}

		/* 4.4 Fa una MPI_Reduce per trovare il massimo globale e aggiorna il massimo e la posizione per
			la tempesta i */
		MPI_Reduce( &local_max, &global_max, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD );
		maximum[i] = global_max.val;
		positions[i] = global_max.pos;
	}

	/* Libera la memoria */
	free(layer);
	free(layer_copy);
}