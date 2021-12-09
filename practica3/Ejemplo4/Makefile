MPICC=mpiicc

CC=icc
CFLAGS=-Wall -std=c99 -O3 -fopenmp

LDFLAGS=-lm

all: heat

heat: heat.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean

run:
	mpirun -np 4 ./heat

clean:
	rm -rf *.o heat heat20000.png

view:
	eog heat20000.png
