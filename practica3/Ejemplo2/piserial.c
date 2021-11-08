#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[])
{
	double pi, exactpi;
	double x, f, area;
	int N;

	if (argc!=2) {
		printf("Execution: ./exec N\n");
		exit(-1);
	} else N = atoi(argv[1]);


	printf("Computing approximation to pi using N = %d\n", N);

	area = 0.0;

	for (int i=0; i<N; i++)
	{
		x = (i+0.5)/N;
		f = 4.0/(1.0 + x*x);
		area = f*(1.0/N); //F(x)*deltax
		pi+=area;
	}


	exactpi = 4.0*atan(1.0);

	printf("pi = %f, %% error = %e\n", pi, fabs(100.0*(pi-exactpi)/exactpi));

	return 0;
}
