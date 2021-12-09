
MPICC=mpiicc

deadlock:
	${MPICC} -o deadlock deadlock.c -g

deadlock_tag:
	${MPICC} -o deadlock_tag deadlock_tag.c -g

all: deadlock_tag deadlock
	echo "compiling"

run:
	mpirun -np 2 ./deadlock

clean:
	rm -f deadlock deadlock_rag *stf*
