#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"


void main(int argc, char* argv[])
{
	double pi, pi_local, exactpi;
	double x, f, area;
	int N;


	if (argc!=2) {
		printf("Execution: ./exec N\n");
		MPI_Finalize();
	} else N = atoi(argv[1]);

	// Here N it's the number of process


	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if (world_rank==0)
		printf("Computing approximation to pi using N = %d\n", N);

	pi_local = 0.0;

	for (int i=(world_rank*N/world_size); i<=((world_rank+1)*N/world_size)-1 ; i++)
	{
		x = (i+0.5)/N;
		f = 4.0/(1.0 + x*x);
		area = f*(1.0/N); //F(x)*deltax
		pi_local += area;
	}

	MPI_Reduce(&pi_local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (world_rank==0){
		exactpi = 4.0*atan(1.0);

		printf("pi = %f, %% error = %e\n", pi, fabs(100.0*(pi-exactpi)/exactpi));
	}

	// Finalize the MPI environment.
	MPI_Finalize();
}
