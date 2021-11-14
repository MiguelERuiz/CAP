//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
//
// Copyright 2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// ============================================================= 


#include <omp.h>

#include <assert.h>
#include <stdio.h>
#include "multiply.h"

#define xstr(s) x_str(s)
#define x_str(s) #s


//=========================================================================================
// OpenMP threading model
//=========================================================================================


void ParallelMultiply(int msize, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int NTHREADS = MAXTHREADS;
	int MSIZE = NUM;

	GetModelParams(&NTHREADS, &MSIZE, 0);

	MULTIPLY(MSIZE, NTHREADS, 0, a, b, c, t);
}

void GetModelParams(int* p_nthreads, int* p_msize, int print)
{
	int msize = NUM;
	int nthr = MAXTHREADS;
	int ncpu = omp_get_max_threads();
	if (ncpu < nthr)
		nthr = ncpu;
	omp_set_num_threads(nthr);

	if(p_nthreads != 0)
		*p_nthreads = nthr;
	if(p_msize != 0)
		*p_msize = msize;

	if(print)
	{
		printf("Threads #: %d %s\n",nthr,"OpenMP threads");	fflush(stdout);
		printf("Matrix size: %d\n",msize); fflush(stdout);
		printf("Using multiply kernel: %s\n", xstr(MULTIPLY)); fflush(stdout);
	}
}

