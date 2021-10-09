/* icc -o fooO0 fooO0.c -O0 -qopt-report=3 -xhost -lm */
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>


double get_time_milisec(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}


void foo( float* sth, float* theta, int p) {

	for(int i = 0; i < p; i++) 
		sth[i] = sin(theta[i]+3.1415927);
}

float reduction(float *a, int p)
{
	float red;
	for(int i = 0; i < p; i++) 
		red+=a[i];

	return(red);
}

int main(int argc, char **argv)
{
	int i;
	int n;
	float *theta, *sth;
	double t0, t1;

	if (argc>1)
		n = atoi(argv[1]);
	else {
		printf("./exec n\n");
		return(-1);
	}

	theta = (float *)malloc(n*sizeof(float));
	sth   = (float *)malloc(n*sizeof(float));

	for (i=0; i<n; i++) {
		theta[i] = (float)(random()%(1000)-500)/1000;
	}

	t0 = get_time_milisec();
	foo(sth, theta, n);
	t1 = get_time_milisec();
	printf("Tiempo %f (ms) red=%f\n", t1-t0, reduction(sth, n)); 


	return(1);
}


