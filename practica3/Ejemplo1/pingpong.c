#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void ping(int N) {
	MPI_Status status;

	double *buffer_send = (double*)malloc(N*sizeof(double));
	double *buffer_recv = (double*)malloc(N*sizeof(double));


	for (int i=0; i<N; i++) buffer_send[i]=i;
	int tag = 1;

	double t1 = MPI_Wtime();
	// dest = 1 -> slave process
	MPI_Send(buffer_send, N, MPI_DOUBLE, 1, tag, MPI_COMM_WORLD);
	MPI_Recv(buffer_recv, N, MPI_DOUBLE, 1, tag, MPI_COMM_WORLD, &status);
	double t2 = MPI_Wtime();

	for (int i=0; i<N; i++)
		if (buffer_send[i]!=buffer_recv[i]) {
			printf("ERROR in COMM: buffers differs\n");
			break;
		}

	printf("Ping pong done in %f secs.\n", t2-t1);

	free(buffer_send);
	free(buffer_recv);
}


void pong(int N) {
	MPI_Status status;
	double *buffer = (double*)malloc(N*sizeof(double));

	// dest = 0 -> master process
	MPI_Recv(buffer, N, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
	MPI_Send(buffer, N, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);

	free(buffer);
}

//--------------------------------------------------------------------
int main( int argc, char *argv[] ) {
	int rank;
	int size;
	int N;

	setbuf (stdout, NULL);

	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc!=2) {
		printf("Execution: ./exec N\n");
		MPI_Finalize();
	} else N = atoi(argv[1]);

	if (rank == 0) ping(N);
	else pong(N);

	MPI_Finalize();

	return 0;
}
