NAME=nbody

CC=icc

CFLAGS=-O2 -g -qopt-report=5 -xCORE-AVX2 -I.

DEPS = gtime.h memory.h nbody.h nbody_routines.h

OBJ = gtime.o memory.o nbody.o nbody_routines.o

LIB = -lm

PRJ = "./Ejemplo5-Report"

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(NAME): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIB)

all: clean $(NAME) survey roofline open-gui

clean:
	rm -rf *.o *.optrpt $(NAME) $(PRJ)

survey:
	advixe-cl -collect survey -project-dir $(PRJ) -- ./$(NAME) 10000

roofline:
	advixe-cl -collect survey -project-dir $(PRJ) -- ./$(NAME) 10000
	advixe-cl -collect tripcounts -flop -project-dir $(PRJ) -- ./$(NAME) 10000

open-gui:
	advixe-gui $(PRJ)/$(PRJ).advixeproj >/dev/null 2>&1 &