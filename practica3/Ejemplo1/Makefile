EXECS=pingpong
MPICC=mpiicc

N=100000000

all:
	${MPICC} -o ${EXECS} pingpong.c

run:
	mpirun -np 2 ./${EXECS} ${N}

clean:
	rm -f ${EXECS}
