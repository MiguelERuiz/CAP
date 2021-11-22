#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef _OPENACC
#include <openacc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static struct timeval time;
double get_time()
{
	double t;
	gettimeofday(&time, (struct timezone*)0);
	t = ((time.tv_usec) + (time.tv_sec)*1000000);

	return (t);
}


int main (int argc, const char *argv[])
{
	int i;
	int n;
	float a = 2.0f;
	float b = 3.0f;

	float *x_acc;
	float *y_acc;
	float *x;
	float *y;

	double t0, t1;

	if (argc!=2)
	{
		printf("./exec n\n");
		return(-1);
	} else
		n = atoi(argv[1]);

#ifdef _OPENMP
	printf("devices avaliables=%d\n", omp_get_num_devices());
#endif

	x_acc = (float*)malloc(n*sizeof(float));
	y_acc = (float*)malloc(n*sizeof(float));
	x     = (float*)malloc(n*sizeof(float));
	y     = (float*)malloc(n*sizeof(float));

	// Init
	for (i=0; i<n; i++){
		x[i] = x_acc[i] = (float)(i);
		y[i] = y_acc[i] = (float)(n-i);
	}
	// SAXPY
	t0 = get_time();

#ifdef _OPENACC
	#pragma acc ....
#endif
#ifdef _OPENMP
	#pragma omp ....
#endif
	for(i=0; i<n; i++)
		y_acc[i] = a*x_acc[i] + y_acc[i];


	t1 = get_time();
	printf("AXPY ACC %f ms\n", (t1-t0)/1000);

	t0 = get_time();
	for(i=0; i<n; i++)
		y[i] = a*x[i] + y[i];

	t1 = get_time();
	printf("AXPY Seq %f ms\n", (t1-t0)/1000);

	// Checking
	for(i=0; i<n; i++)
		if(y_acc[i]!=y[i])
			printf("!axpy in %i (%f!=%f)\n", i, y_acc[i], y[i]);

	free(x); free(y);
	free(x_acc); free(y_acc);
	return(0);
}

