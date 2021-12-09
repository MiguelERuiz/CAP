EXECS=hello
MPICC=mpicc

all:
	${MPICC} -o ${EXECS} hello.c

run:
	mpirun -np 4 ./${EXECS}

clean:
	rm -f ${EXECS}
