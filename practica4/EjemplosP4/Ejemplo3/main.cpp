#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <omp.h>
#include "mandelbrot.h"
#include "constants.h"

#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;

int main() {

  size_t bytes=WIDTH*HEIGHT*sizeof(unsigned int);
  unsigned char *image=(unsigned char*)malloc(bytes);

  //int num_blocks, block_size;
  FILE *fp=fopen("image.pgm","wb");
  fprintf(fp,"P5\n%s\n%d %d\n%d\n","#comment",WIDTH,HEIGHT,MAX_COLOR);

  // This region absorbs overheads that occur once in a typical run
  // to prevent them from skewing the results of the example.
  image[0] = 0;
  double st = omp_get_wtime();
  int numBlocks = (HEIGHT/SIZE_BLOCK);

  #ifdef _OPENACC
  #pragma acc data copy(image[0:WIDTH*HEIGHT])
  #endif
  #ifdef _OPENMP
  #pragma omp target data map(tofrom: image[0:WIDTH*HEIGHT])
  #endif
  for(int block=0; block < numBlocks; block++){
      int start = block * SIZE_BLOCK;
      int end = start + SIZE_BLOCK;

      #ifdef _OPENACC
        #pragma acc parallel loop
      #endif
      #ifdef _OPENMP
      #pragma omp target teams distribute parallel for simd collapse(2)
      #endif
      for(int y=start;y<end;y++) {
        for(int x=0;x<WIDTH;x++) {
            image[y*WIDTH+x]=mandelbrot(x,y);
        }
      }


  }

  double et = omp_get_wtime();
  printf("Time: %lf seconds.\n", (et-st));
  fwrite(image,sizeof(unsigned char),WIDTH*HEIGHT,fp);
  fclose(fp);
  free(image);
  return 0;
}
