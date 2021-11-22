
ICX=icx

CFLAGS=-O3 -lm -lrt -std=c99

PGCC=pgcc -Minfo -fast -acc -ta=nvidia -tp=nehalem

CFLAGS_OFFLOAD = $(CFLAGS) -fiopenmp -fopenmp-targets=spir64 -qopt-report


NAME=hello

default: host gpu

host:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).host.exe

gpu: pgi omp_offload

pgi:
	$(PGCC) $(NAME).c -o $(NAME).pgi.exe

omp_offload:
	$(ICX) $(CFLAGS_OFFLOAD) $(NAME).c -o $(NAME).icx.exe

clean: 
	rm -f __hmpp_*  *.exe *.yaml
