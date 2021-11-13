CC=icc
CFLAGS=-O3 -std=c99 -Wall -fopenmp -qopt-report=5 -g
LDFLAGS=


TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o
FUNCTION_TARGET=lin_solve

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

collect-hotspots:
	vtune -collect hotspots -app-working-dir . -r ./hotspots-collections \
	-source-search-dir=/home/usercap04/CAP/practica2/Ejemplo7 \
	-- ./headless > hotspots-collect

collect-threading:
	vtune -collect threading -app-working-dir . -r ./threading-collections \
	-source-search-dir=/home/usercap04/CAP/practica2/Ejemplo7 \
	-- ./headless > threading-collect

# Exectue after collect-hostspots
report-hotspots:
	vtune -report hotspots -r hotspots-collections -source-object function=$(FUNCTION_TARGET) \
	-report-output=source-hotspots \
	-source-search-dir=/home/usercap04/CAP/practica2/Ejemplo7

clean:
	rm -rf $(TARGETS) *.o *.optrpt .depend *~ r00*/ ./*-collections/ \
	./*-results *-report source-*

.depend: *.[ch]
	$(CC) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
