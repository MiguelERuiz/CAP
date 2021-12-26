CXX=c++
CXXFLAGS=-O3 -fopenmp

ICX=icpx
PGCC=nvc++ -Minfo -fast -acc -ta=nvidia -tp=nehalem
CFLAGS_OFFLOAD = $(CXXFLAGS) -fiopenmp -fopenmp-targets=spir64 -qopt-report

default: host gpu

NAME=mandelbrot


host: mandelbrot.o main.o
	$(CXX) $(CXXFLAGS) -o $(NAME).host.exe $^

gpu: omp_offload pgi

pgi:
	$(PGCC) -c mandelbrot.cpp
	$(PGCC) -c main.cpp
	$(PGCC) -o $(NAME).pgi.exe mandelbrot.o main.o $(LIB)

omp_offload:
	$(ICX) $(CFLAGS_OFFLOAD) -c mandelbrot.cpp
	$(ICX) $(CFLAGS_OFFLOAD) -c main.cpp
	$(ICX) $(CFLAGS_OFFLOAD) -o $(NAME).icx.exe mandelbrot.o main.o


clean:
	rm -f *.o __hmpp_*  *.exe *.yaml image.pgm
