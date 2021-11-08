EXECS=hello
MPICC=mpicc

all:
	${MPICC} -o ${EXECS} hello.c

run:
	mpirun -np 2 ./${EXECS}

clean:
	rm -f ${EXECS}
