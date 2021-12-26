#include <stdio.h>

#ifdef _OPENACC
#include <openacc.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define N 1000

int main() {

	int a[N];
	int b[N];

#ifdef _OPENACC
	acc_init(acc_device_not_host);
	printf(" Compiling with OpenACC support \n");
#endif 

#ifdef _OPENMP
	printf("There are %d devices\n", omp_get_num_devices());
#endif 

	printf(" Hello World! \n ");

	// Compute on the host
	for (int i = 0; i < N; i++) {
		a[i] = i;
	}
	// Compute on the GPU if OpenACC/OpenMP support - host if not
#ifdef _OPENACC
   #pragma acc kernels copy(b[0:N])
#endif
#ifdef _OPENMP
   #pragma omp target map(from:b[0:N])
   #pragma omp parallel for
#endif
	for (int i = 0; i < N; i++) {
		b[i] = i;
	}

	for (int i = 0; i < N; i++) {
		if (a[i] != b[i])  {
			printf("Something went wrong\n");
		return 1;
		}
	}
	printf("Done\n");
#ifdef _OPENACC
	acc_shutdown(acc_device_not_host);
#endif 

   return 0;

}
