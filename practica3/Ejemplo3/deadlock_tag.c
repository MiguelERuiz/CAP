#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"


//--------------------------------------------------------------------
int main( int argc, char *argv[] ) {
	int rank;
	int size;

	MPI_Status status;
	int tag1=1, tag2=2;
	int length=100;
	char buffer1[100], buffer2[100];
	
	setbuf (stdout, NULL);
	
	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (argc!=1) {
		printf("Execution: ./exec\n");
		MPI_Finalize();
		return(-1);
	}
	
	/* Call ping-pong functions */
	if (rank == 0) {
		sprintf(buffer1, "process %d of %d. Sending..\n", rank, size); 
		MPI_Ssend(buffer1, length, MPI_FLOAT, 1, tag1,
			MPI_COMM_WORLD);

		MPI_Recv(buffer2, length, MPI_FLOAT, 1, tag2,
			MPI_COMM_WORLD, &status);

		printf("pingpong performed\n");

		MPI_Ssend(buffer1, length, MPI_FLOAT, 1, tag1,
			MPI_COMM_WORLD);

		MPI_Ssend(buffer2, length, MPI_FLOAT, 1, tag2,
			MPI_COMM_WORLD);
	} else if (rank == 1) {
		sprintf(buffer2, "process %d of %d. Sending...\n", rank, size); 
		MPI_Recv(buffer1, length, MPI_FLOAT, 0, tag1,
			MPI_COMM_WORLD, &status);

		MPI_Ssend(buffer2, length, MPI_FLOAT, 0, tag2,
			MPI_COMM_WORLD);

		MPI_Recv(buffer1, length, MPI_FLOAT, 0, tag2,
			MPI_COMM_WORLD, &status);

		MPI_Recv(buffer2, length, MPI_FLOAT, 0, tag1,
			MPI_COMM_WORLD, &status);
	}
	
	MPI_Finalize();
	
	return 0;
}
