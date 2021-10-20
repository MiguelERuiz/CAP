#include <stdio.h>
#include <math.h>
#include <omp.h>

double f(double x);    /* Function we're integrating */
double Trap(double a, double b, int n, double h);

int main(int argc, char **argv) {
	double  integral;	/* Store result in integral   */
	double  a=0.0, b=1.0;	/* Left and right endpoints   */
	int     n=10000000;	/* Number of trapezoids       */
	double  h;		/* Height of trapezoids       */

	if (argc==2){
		n = atoi(argv[1]);
	} else if(argc>2){
		printf("./exec num_traps\n");
		exit(-1);
	}
	

	h = (b-a)/n;
	double start = omp_get_wtime();
	integral = Trap(a, b, n, h);
	double stop = omp_get_wtime();

	printf("With n = %d trapezoids\n", n);
	printf("Integral from %f to %f = %.15f\n",
		a, b, integral);
	printf("Computed in %f s.\n", stop-start);

	return 0;
}  /* main */

/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule anqd
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double Trap(double a, double b, int n, double h) {
	double integral, area;
	int k;

	integral = 0.0;
	for (k = 1; k <= n; k++) {
		area = h*(f(a+k*h)+f(a+(k-1)*h))/2.0;
		integral+=area;
	}

	return integral;
}  /* Trap */

/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double f(double x) {
	double return_val;

	return_val = 1.0/(1.0+exp(x*x-4*x-10.0)+sin(x/3.14));
	return return_val;
}  /* f */
