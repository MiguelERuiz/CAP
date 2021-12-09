#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"


//--------------------------------------------------------------------
int main( int argc, char *argv[] ) {
	int rank;
	int size;
	int N;

	MPI_Status status;
	int tag;
	int other;
	char message[100];
	int sendfirst=1;
	
	setbuf (stdout, NULL);
	
	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	other = (rank+1)%2;
	
	if (argc!=2) {
		printf("Execution: ./exec 0/1\n");
		MPI_Finalize();
		return(-1);
	} else sendfirst=atoi(argv[1]);
	
	/* now for the comms */
	if (sendfirst == 1) {
		printf("process %d of %d. Sending..\n",rank,size);
		sprintf(message, "process %d of %d. Sending..\n",rank,size); 
		/* first send.. */
		MPI_Ssend(message, BUFSIZ, MPI_CHAR, other, tag, MPI_COMM_WORLD);
		/* ..then receive */ 
		MPI_Recv(message, BUFSIZ, MPI_CHAR, other, tag ,MPI_COMM_WORLD, &status);
		/* 
		** We _may_ get here.. 
		** but only if the blocking function MPI_Send() has access to a system buffer
		*/
		printf("process %d of %d. Received.\n",rank,size);  
	} else {
		printf("process %d of %d. Receiving..\n",rank,size);
		MPI_Recv(message, BUFSIZ, MPI_CHAR, other, tag, MPI_COMM_WORLD, &status);

		MPI_Ssend(message, BUFSIZ ,MPI_CHAR, other, tag, MPI_COMM_WORLD);
		/* 
		** The blocking function MPI_Recv() cannot return and so we cannot get here..
		*/ 
		printf("We'll never get here!\n");
	}
	
	MPI_Finalize();
	
	return 0;
}
