CC   = icc

EXE   = main

SOURCES    = main.c

PRJ = "./Ejemplo7-Report"

OBJS    = stencil.o pngio.o main.o

CFLAGS     = -O2 -g -qopt-report=5 -xHost
NVCFLAGS   = $(CFLAGS)

LIBS =  -lm -lpng

SOURCEDIR = .

$(EXE) :$(OBJS)
	$(CC) $(CFLAGS)  -o $@ $? -I. $(LIBS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $< -I.
clean:
	rm -rf *.optrpt $(OBJS) $(EXE) $(PRJ)

survey:
	advixe-cl -collect survey -project-dir $(PRJ) -- ./$(EXE) test-image.png

# roofline:
# 	advixe-cl -collect survey -project-dir $(PRJ) -- ./$(EXE) test-image.png
# 	advixe-cl -collect tripcounts -flop -project-dir $(PRJ) -- ./$(EXE) test-image.png

open-gui:
	advixe-gui $(PRJ)/$(PRJ).advixeproj >/dev/null 2>&1 &
