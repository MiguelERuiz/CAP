
CC=icc

CFLAGS=-O2 -g -qopt-report=3 -xhost

LIB=-lm -qopenmp

NAME=trapezoidal

default: serial

serial:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).serial.exe $(LIB)

clean:
	rm -rf *exe *.o *.out *.optrpt
