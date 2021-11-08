CC=gcc
CFLAGS=-Wall -std=c99 -O3 -fopenmp

LDFLAGS=-lm

all: heat

heat: heat.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf *.o heat
