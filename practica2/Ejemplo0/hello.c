/* To compile: icc -o hello hello.c -qopenmp */
/* To execute: export OMP_NUM_THREADS=4; ./hello */
#include <stdio.h>
#include <omp.h>

int main(){

	// This code is executed by 1 thread
	printf("OpenMP with %d max threads\n", omp_get_max_threads () );
	printf("OpenMP with %d procs available\n", omp_get_num_procs () );

	#pragma omp parallel
	{
	// This code is executed in parallel
	// by multiple threads
	printf("Hello World from thread %d of %d threads\n",
		omp_get_thread_num(), omp_get_num_threads() );
	}
}

