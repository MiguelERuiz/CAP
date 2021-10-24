#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include "stencil.h"
#include "pngio.h"


/* Time */
#include <sys/time.h>
#include <sys/resource.h>

double get_time_sec(){
	static struct timeval 	tv0;
	double time_, mytime;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	mytime = time_/1000000;
	return(mytime);
}


const int nTrials = 2;
const int skipTrials = 1; // Skip first iteration as warm-up

int main(int argc, char **argv) {

	unsigned char *img_in, *img_out, *img_name_out;
	int width, height;

	if(argc < 2) {
		printf("Usage: %s {file} {out}\n", argv[0]);
		exit(1);
	}

	img_in  = readImage(argv[1], &width, &height);
	img_out = (unsigned char*)_mm_malloc(sizeof(unsigned char)* width * height, 16);

	printf("\n\033[1mEdge detection with a 3x3 stencil\033[0m\n");
	printf("\nImage size: %d x %d\n\n", width, height);
	printf("\033[1m%5s %15s %15s %15s\033[0m\n", "Step", "Time, ms", "GB/s", "GFLOP/s"); fflush(stdout);

	double t, dt, g, dg, f, df;

	for (int iTrial = 1; iTrial <= nTrials; iTrial++) {
		const double t0 = get_time_sec();
		ApplyStencil(img_in, img_out, width, height);
		const double t1 = get_time_sec();

		const double ts   = t1-t0; // time in seconds
		const double tms  = ts*1.0e3; // time in milliseconds
		const double gbps = sizeof(unsigned char)*width*height*2*1e-9/ts; // bandwidth in GB/s
		const double fpps = width*height*2*9*1e-9/ts; // performance in GFLOP/s

		if (iTrial > skipTrials) { // Collect statistics
			t  += tms;
			dt += tms*tms;
			g  += gbps;
			dg += gbps*gbps;
			f  += fpps;
			df += fpps*fpps;
		}

		// Output performance
		printf("%5d %15.3f %15.3f %15.3f %s\n",
		iTrial, tms, gbps, fpps, (iTrial<=skipTrials?"*":""));
		fflush(stdout);

		// Copy image
		CopyImage(img_in, img_out, width, height);
	}


	writeImage("test_out.png", img_out, width, height);

	_mm_free(img_in);
	_mm_free(img_out);

	return(0);
}

