
MPICC=mpiicc
CC=icc
CFLAGS=-O2 -lm

all:
	$(CC)    -o piserial piserial.c $(CFLAGS)
	$(MPICC) -o piparallel piparallel.c $(CFLAGS)

run:
	mpirun -np 2 ./piparallel 100000000000000

clean:
	rm -f piserial piparallel
