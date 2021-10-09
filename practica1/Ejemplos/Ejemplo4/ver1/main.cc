#include <cstdio>
#include <omp.h>
#include <cmath>

#include "cpu_time.hpp"

struct Charge { // Elegant, but ineffective data layout
    float x, y, z, q; // Coordinates and value of this charge
    int type; // Positive or negative charge
};

// This version performs poorly, because data layout of class Charge 
// does not allow efficient vectorization
void CalculateElectricPotential(
        const int m,       // Number of charges
        const Charge* chg, // Charge distribution (array of classes)
        const float Rx, const float Ry, const float Rz, // Observation point
        float & phi  // Output: electric potential
        ) {
    phi=0.0f;
    for (int i=0; i<m; i++)  { // This loop will be auto-vectorized
        // Non-unit stride: (&chg[i+1].x - &chg[i].x) == sizeof(Charge)
        const float dx=chg[i].x - Rx;
        const float dy=chg[i].y - Ry;
        const float dz=chg[i].z - Rz;
        phi -= chg[i].q / sqrtf(dx*dx+dy*dy+dz*dz); // Coulomb's law
    }
}


float rngNumber(float minV, float maxV, float precision)
{
	float range = (maxV-minV)*precision;
	int irange = range;

	return((rand()%irange)/precision+minV);
}


int main(int argv, char* argc[]){
    const size_t n=1<<10;
    const size_t m=1<<10;
    const int nTrials=10;
    const int skipTrials=2;

    CPUTime time;

    Charge chg[m];
    float* potential = (float*) malloc(sizeof(float)*n*n);

    // Initializing array of charges
    printf("Initialization...");

    for (size_t i=0; i<n; i++) {
      chg[i].x = rngNumber(-5.0, 5.0, 1000);
      chg[i].y = rngNumber(-5.0, 5.0, 1000);
      chg[i].z = rngNumber(-5.0, 5.0, 1000);
      chg[i].q = rngNumber(-5.0, 5.0, 1000);
    }
    printf(" complete.\n");

    printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GFLOP/s");
    double perf=0.0, dperf=0.0;
    for (int t=1; t<=nTrials; t++){
        
	for (int i=0; i<n*n; i++)
		potential[i]=0.0f;

        const double t0 = time.start();

        for (int j = 0; j < n*n; j++) {
            const float Rx = (float)(j % n);
            const float Ry = (float)(j / n);
            const float Rz = 0.0f;
            CalculateElectricPotential(m, chg, Rx, Ry, Rz, potential[j]);
        }
        const double t1 = time.start();

	const double HztoPerf = 10.0*1e-9*double(n*n)*double(m);
	if (t > skipTrials) {
	  perf += HztoPerf/(t1-t0);
	  dperf += HztoPerf*HztoPerf/((t1-t0)*(t1-t0));
	}

	printf("%5d %10.3e %8.1f %s\n", 
	       t, (t1-t0), HztoPerf/(t1-t0), (t<=skipTrials?"*":""));
	fflush(stdout);
    }
    perf/=(double)(nTrials-skipTrials); 
    dperf=sqrt(dperf/(double)(nTrials-skipTrials)-perf*perf);
    printf("-----------------------------------------------------\n");
    printf("\033[1m%s %4s \033[42m%10.1f +- %.1f GFLOP/s\033[0m\n",
	   "Average performance:", "", perf, dperf);
    printf("-----------------------------------------------------\n");
    printf("* - warm-up, not included in average\n\n");
    free(potential);
}
