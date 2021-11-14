#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define DEBUG 1

void bubble_sort(int list[], long n)
{
	long c, d, t;

	//...
}

void main(int argc, char**argv)
{
	int n;
	int i, j, k=0;
	int not_flag;
	int *primes;

	if (argc<2)
	{
		printf("./exec num\n");
		exit(-1);
	} else
		n = atoi(argv[1]);

	primes = (int *)malloc(sizeof(int)*n/2+1);

	double start = omp_get_wtime();

	//#pragma omp parallel...
	for (i=2; i<n; i++)
	{
		not_flag = 0;
		j=2;
		while(j*j<=i && !not_flag)
		{
			if(i%j==0) // not prime
				not_flag=1;
			j++;
		}
		if (j*j>=i && !not_flag){
			primes[k++] = i;
		}

	}

	double stop = omp_get_wtime();

	//bubble_sort(primes, k);

#ifdef DEBUG
	printf("List of primes=");
	for (i=0; i<k; i++)
		printf("%d ", primes[i]);
	printf("\n");
#endif
	printf("Number of primes between 1-%d = %d\n", n, k);
	printf("Computed in %f s.\n", stop-start);

	free(primes);

}
