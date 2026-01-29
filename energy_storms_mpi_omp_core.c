#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "energy_storms.h"

/* Funzione per aggiornare una singola posizione del layer */
static void update( float *layer, int layer_size, int k, int pos, float energy ) {
	/* 1. Calcola il valore assoluto della distanza tra la
		posizione di impatto e la k-esima posizione del layer */
	int distance = pos - k;
	if ( distance < 0 ) distance = - distance;

	/* 2. La cella di impatto ha un valore di distanza pari a 1 */
	distance = distance + 1;

	/* 3. Radice quadrata della distanza */
	/* NOTA: L'attenuazione nel mondo reale dipende tipicamente dal quadrato della distanza.
	   Qui usiamo un'equazione personalizzata che influenza un intervallo molto più ampio di celle */
	float atenuacion = sqrtf( (float)distance );

	/* 4. Calcola l'energia attenuata */
	float energy_k = energy / layer_size / atenuacion;

	/* 5. Non aggiungere se il valore assoluto è inferiore alla soglia */
	if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
		layer[k] = layer[k] + energy_k;
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
	int i, j, k, rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* 3. Assegna una porzione del layer ad ogni processo MPI e
		ricalcola la dimensione locale */
	int first_k = (rank * layer_size) / size;
	int last_k = ((rank + 1) * layer_size) / size;
	int local_size = last_k - first_k;

	/* Alloca la memoria e la inizializza a 0, aggiungendo due "ghost cells" usate per lo scambio 
		dei bordi più avanti */
	float *layer = (float *)malloc( sizeof(float) * (local_size + 2) );
	float *layer_copy = (float *)malloc( sizeof(float) * (local_size + 2) );
	if ( layer == NULL || layer_copy == NULL ) {
		fprintf(stderr, "Error: Allocating the layer memory\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	#pragma omp parallel for schedule(static)
	for( k=0; k<local_size+2; k++ ) {
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
	for( i=0; i<num_storms; i++ ) {

		/* 4.1. Aggiungi le energie degli impatti alle celle del layer */
		#pragma omp parallel private(j, k)
		{
			/* Per ogni particella */
			for( j=0; j<storms[i].size; j++ ) {
				/* Ottieni l'energia di impatto */
				float energy = (float)storms[i].posval[j*2+1] * 1000;
				/* Ottieni la posizione di impatto */
				int position = storms[i].posval[j*2];
				/* Calcola la posizione locale  */
				int local_position = position - first_k + 1;

				/* Per ogni cella nel layer */
				#pragma omp for schedule(static) nowait
				for( k=1; k<=local_size; k++ ) {
					/* Aggiorna l'energia */
					update( layer, layer_size, k, local_position, energy );
				}
			}
		}

		/* 4.2. Rilassamento dell'energia tra le tempeste */
		/* 4.2.1. Copia i valori nell'array copia */
		#pragma omp parallel for schedule(static) private(k)
		for( k=1; k<=local_size; k++ ) 
			layer_copy[k] = layer[k];

		/* 4.2.2 Inizia lo scambio dei valori. I processi MPI inviano i propri bordi ai vicini
				per permettere il calcolo del rilassamento ai bordi (che non sono quelli globali) */
		/* Invia la prima cella e riceve l'ultima */
		MPI_Sendrecv(&layer_copy[1], 1, MPI_FLOAT, left_rank, 1,
					&layer_copy[0], 1, MPI_FLOAT, left_rank, 1,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		/* Invia l'ultima cella e riceve la prima */
		MPI_Sendrecv(&layer_copy[local_size], 1, MPI_FLOAT, right_rank, 1,
					&layer_copy[local_size+1], 1, MPI_FLOAT, right_rank, 1,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* 4.2.3. Aggiorna il layer usando i valori della copia. Vengono aggiornati i valori ai bordi
				usando le "ghost cells" allocate all'inizio */
		#pragma omp parallel for schedule(static) private(k)
		for( k=1; k<=local_size; k++ ) {
			layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3;
		}

		/* 4.2.4 Riavviene lo scambio dei valori aggiornati per il calcolo del massimo */
		/* Invia la prima cella e riceve l'ultima */
		MPI_Sendrecv(&layer[1], 1, MPI_FLOAT, left_rank, 2,
					&layer[0], 1, MPI_FLOAT, left_rank, 2,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		/* Invia l'ultima cella e riceve la prima */
		MPI_Sendrecv(&layer[local_size], 1, MPI_FLOAT, right_rank, 2,
					&layer[local_size+1], 1, MPI_FLOAT, right_rank, 2,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* 4.3. Trova il valore massimo nel layer */
		/* Crea una struct per avere il tipo di dato corretto nella MPI_Allreduce */
		struct { float val; int pos; } local_max, global_max;
		local_max.val = 0;
		local_max.pos = 0;

		#pragma omp parallel private(k)
		{
			float thread_max_val = 0;
			int thread_max_pos = 0;

			/* Trova il massimo valore nel layer locale */
			#pragma omp for schedule(static) nowait
			for( k=1; k<=local_size; k++ ) {
				/* Controlla solo se è un massimo locale */
				if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
					if ( layer[k] > thread_max_val ) {
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

		/* 4.4 Fa una MPI_Allreduce per trovare il massimo globale e aggiorna il massimo e la posizione per
			la tempesta i */
		MPI_Allreduce( &local_max, &global_max, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD );
		maximum[i] = global_max.val;
		positions[i] = global_max.pos;
	}

	/* Libera la memoria */
	free( layer );
	free( layer_copy );
}