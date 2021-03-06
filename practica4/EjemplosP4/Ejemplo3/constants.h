#pragma omp declare target
const unsigned int WIDTH=8192;
const unsigned int HEIGHT=8192;
const unsigned int MAX_ITERS=50;
const unsigned int MAX_COLOR=255;
const unsigned int SIZE_BLOCK = 1024;
const double xmin=-1.7;
const double xmax=.5;
const double ymin=-1.2;
const double ymax=1.2;
const double dx=(xmax-xmin)/WIDTH;
const double dy=(ymax-ymin)/HEIGHT;
#pragma omp end declare target
