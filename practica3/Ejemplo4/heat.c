#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "colormap.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Simulation parameters
static const unsigned int N = 500;

static const float SOURCE_TEMP = 5000.0f;
static const float BOUNDARY_TEMP = 1000.0f;

static const float MIN_DELTA = 0.05f;
static const unsigned int MAX_ITERATIONS = 20000;


static unsigned int idx(unsigned int x, unsigned int y, unsigned int stride) {
    return y * stride + x;
}


static void init(unsigned int source_x, unsigned int source_y, float * matrix) {
	// init
	memset(matrix, 0, N * N * sizeof(float));

	// place source
	matrix[idx(source_x, source_y, N)] = SOURCE_TEMP;

	// fill borders
	for (unsigned int x = 0; x < N; ++x) {
		matrix[idx(x, 0,   N)] = BOUNDARY_TEMP;
		matrix[idx(x, N-1, N)] = BOUNDARY_TEMP;
	}
	for (unsigned int y = 0; y < N; ++y) {
		matrix[idx(0,   y, N)] = BOUNDARY_TEMP;
		matrix[idx(N-1, y, N)] = BOUNDARY_TEMP;
	}
}

static void step(unsigned int source_x, unsigned int source_y, const float * current, float * next, int rank, int w_size) {

	float recv_ghost_row[N];
	float send_ghost_row[N];
	float recv_ghost_row1[N];
	float send_ghost_row1[N];
	unsigned int slice = (N/w_size);
	unsigned int last_row = (slice*rank)+(slice-1);

	MPI_Status status;

	if (rank == 0)
	{
		for (unsigned int y = 1; y < slice-1; ++y) {
			for (unsigned int x = 1; x < N-1; ++x) {

				if ((y == source_y) && (x == source_x)) {
					continue;
				}
				next[idx(x, y, N)] = (current[idx(x, y-1, N)] +
															current[idx(x-1, y, N)] +
															current[idx(x+1, y, N)] +
															current[idx(x, y+1, N)]) / 4.0f;
			}
		}
		// Recibe la fila ghost del proceso remoto
		MPI_Recv(recv_ghost_row, N, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status);

		// Se procesa la última fila
		for (unsigned int x = 1, y = last_row; x < N - 1; ++x)
		{
			if ((y == source_y) && (x == source_x)) {
					continue;
				}
			next[idx(x, y, N)] = (current[idx(x, y-1, N)] +
															current[idx(x-1, y, N)] +
															current[idx(x+1, y, N)] +
															recv_ghost_row[x]) / 4.0f;
		}

		for (unsigned int x = 0; x < N - 1; ++x)
		{
			send_ghost_row[x] = current[idx(x, last_row, N)];
		}

		MPI_Send( send_ghost_row , N , MPI_FLOAT , 1 , 0 , MPI_COMM_WORLD);

	} else if(0 < rank && rank < w_size-1) {
		int former = rank - 1;
		int latter = rank + 1;
		// Enviar y recibir con su anterior

		// Enviar y recibir con su posterior
		MPI_Recv(recv_ghost_row1, N, MPI_FLOAT, latter, latter, MPI_COMM_WORLD, &status);

		for (unsigned int x = 1, y = last_row; x < N - 1; ++x)
		{
			if ((y == source_y) && (x == source_x)) {
					continue;
			}
			next[idx(x, y, N)] = (current[idx(x, y-1, N)] +
															current[idx(x-1, y, N)] +
															current[idx(x+1, y, N)] +
															recv_ghost_row1[x]) / 4.0f;
		}

		for (unsigned int x = 0; x < N - 1; ++x)
		{
			send_ghost_row1[x] = current[idx(x, last_row, N)];
		}

		MPI_Send( send_ghost_row1 , N , MPI_FLOAT , latter , rank , MPI_COMM_WORLD);
		for (unsigned int x = 0; x < N; ++x)
		{
			send_ghost_row[x] = current[idx(x, slice*rank, N)];
		}
		MPI_Send( send_ghost_row , N , MPI_FLOAT , former , rank , MPI_COMM_WORLD);

		// Recibo de la fila ghost del proceso anterior
		MPI_Recv( recv_ghost_row , N , MPI_FLOAT , former , former , MPI_COMM_WORLD , &status);

		for (unsigned int x = 1, y = slice * rank; x < N - 1; ++x)
			{
				if ((y == source_y) && (x == source_x)) {
					continue;
				}
				next[idx(x, y, N)] = (recv_ghost_row[x] +
															current[idx(x-1, y, N)] +
															current[idx(x+1, y, N)] +
															current[idx(x, y+1, N)]) / 4.0f;
			}

		for (unsigned int y = slice*rank+1; y < (rank*slice)+(slice-1); ++y) {
			for (unsigned int x = 1; x < N-1; ++x) {

				if ((y == source_y) && (x == source_x)) {
					continue;
				}
				next[idx(x, y, N)] = (current[idx(x, y-1, N)] +
															current[idx(x-1, y, N)] +
															current[idx(x+1, y, N)] +
															current[idx(x, y+1, N)]) / 4.0f;
			}
		}
	} else {
		// Envio de la fila ghost para el proceso maestro
		for (unsigned int x = 0; x < N; ++x)
		{
			send_ghost_row[x] = current[idx(x, slice*rank, N)];
		}
		MPI_Send( send_ghost_row , N , MPI_FLOAT , rank-1 , rank , MPI_COMM_WORLD);
		// Recibo de la fila ghost del proceso maestro
		MPI_Recv( recv_ghost_row , N , MPI_FLOAT , rank-1 , rank-1 , MPI_COMM_WORLD , &status);
		for (unsigned int x = 1, y = slice*rank; x < N - 1; ++x)
		{
			if ((y == source_y) && (x == source_x)) {
				continue;
			}
			next[idx(x, y, N)] = (recv_ghost_row[x] +
														current[idx(x-1, y, N)] +
														current[idx(x+1, y, N)] +
														current[idx(x, y+1, N)]) / 4.0f;
		}
		for (unsigned int y = slice*rank + 1; y < N-1; ++y) {
			for (unsigned int x = 1; x < N-1; ++x) {
				if ((y == source_y) && (x == source_x)) {
					continue;
				}
				next[idx(x, y, N)] = (current[idx(x, y-1, N)] +
															current[idx(x-1, y, N)] +
															current[idx(x+1, y, N)] +
															current[idx(x, y+1, N)]) / 4.0f;
			}
		}
	}
}

static float diff(const float * current, const float * next, int rank, int w_size) {
	int slice = (N/w_size) ;
	float maxdiff = 0.0f;
	if (rank == 0)
	{
		for (unsigned int y = 1; y < slice; ++y) {
			for (unsigned int x = 1; x < N-1; ++x) {
				maxdiff = fmaxf(maxdiff, fabsf(next[idx(x, y, N)] - current[idx(x, y, N)]));
			}
		}
	} else if(0 < rank && rank < w_size) {
		for (unsigned int y = 1; y < slice * rank+slice; ++y) {
			for (unsigned int x = 1; x < N-1; ++x) {
				maxdiff = fmaxf(maxdiff, fabsf(next[idx(x, y, N)] - current[idx(x, y, N)]));
			}
		}
	} else {
		for (unsigned int y = slice; y < N-1; ++y) {
			for (unsigned int x = 1; x < N-1; ++x) {
				maxdiff = fmaxf(maxdiff, fabsf(next[idx(x, y, N)] - current[idx(x, y, N)]));
			}
		}
	}

	return maxdiff;
}


void write_png(float * current, int iter) {
	char file[100];
	uint8_t * image = malloc(3 * N * N * sizeof(uint8_t));
	float maxval = fmaxf(SOURCE_TEMP, BOUNDARY_TEMP);

	for (unsigned int y = 0; y < N; ++y) {
		for (unsigned int x = 0; x < N; ++x) {
			unsigned int i = idx(x, y, N);
			colormap_rgb(COLORMAP_MAGMA, current[i], 0.0f, maxval, &image[3*i], &image[3*i + 1], &image[3*i + 2]);
		}
	}
	sprintf(file,"heat%i.png", iter);
	stbi_write_png(file, N, N, 3, image, 3 * N);

	free(image);
}


int main() {

	MPI_Status status;

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	size_t array_size = N * N * sizeof(float);

	float * current = malloc(array_size);
	float * next = malloc(array_size);

	srand(0);
	unsigned int source_x = rand() % (N-2) + 1;
	unsigned int source_y = rand() % (N-2) + 1;
	printf("Heat source at (%u, %u)\n", source_x, source_y);

	init(source_x, source_y, current);
	memcpy(next, current, array_size);

	double start = MPI_Wtime();

	float t_diff = SOURCE_TEMP;
	float global_t_diff = SOURCE_TEMP;

	for (unsigned int it = 0; (it < MAX_ITERATIONS) && (global_t_diff > MIN_DELTA); ++it) {
		step(source_x, source_y, current, next, world_rank, world_size);
		t_diff = diff(current, next, world_rank, world_size);
		MPI_Allreduce(&t_diff, &global_t_diff, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		if(it%(MAX_ITERATIONS/10)==0 && world_rank == 0){
			printf("%u: %f\n", it, global_t_diff);
		}

		float * swap = current;
		current = next;
		next = swap;
	}

	if(world_rank==0){

		unsigned int slice = (N/world_size);
		double stop = MPI_Wtime();

		printf("Computing time %f s.\n", stop-start);

		float * aux = malloc(array_size);

		for (unsigned int i = 1; i < world_size; i++)
		{
			MPI_Recv( aux , N*N , MPI_FLOAT , i , 1 , MPI_COMM_WORLD , &status);
			for (unsigned int y = slice*i; y < (i == (world_size-1) ? N-1 : (slice*i)+slice); ++y) {
				for (unsigned int x = 0; x < N; ++x) {
					current[idx(x, y, N)] = aux[idx(x, y, N)];
				}
			}
		}

		write_png(current, MAX_ITERATIONS);

		free(aux);

	} else
	{
		MPI_Send(current, N * N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
	}



	free(current);
	free(next);

	// Finalize the MPI environment.
	MPI_Finalize();

	return 0;
}
