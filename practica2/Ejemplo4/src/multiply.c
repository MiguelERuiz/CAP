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

// matrix multiply routines
#include "multiply.h"

#include "mkl.h"


void multiply0(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
	int i,j,k;

// Basic serial implementation
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
    	    for(k=0; k<msize; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	} 
}

void multiply1(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{ 
	int i,j,k;

	// Basic parallel implementation
	#pragma omp parallel for
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
    	    for(k=0; k<msize; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	} 
}

void multiply2(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k;

	// Parallel with merged outer loops
	#pragma omp parallel for collapse (2)
    for(i=0; i<msize; i++) {
        for(j=0; j<msize; j++) {
    	    for(k=0; k<msize; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	} 
}
void multiply3(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k;

	#pragma omp parallel for collapse (2)
	for(i=0; i<msize; i++) {
		for(k=0; k<msize; k++) {
#pragma ivdep
			for(j=0; j<msize; j++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	}
}
void multiply4(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{
	int i,j,k;

	#pragma omp parallel for collapse (2)
	for(i=0; i<msize; i++) {
		for(k=0; k<msize; k++) {
#pragma unroll(8)
#pragma ivdep
			for(j=0; j<msize; j++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	}
}



// DGEMM way of matrix multiply using Intel MKL
// Link with Intel MKL library: With MSFT VS and Intel Composer integration: Select build components in the Project context menu.
// For command line - check out the Intel® Math Kernel Library Link Line Advisor 
void multiply5(int msize, int tidx, int numt, TYPE a[][NUM], TYPE b[][NUM], TYPE c[][NUM], TYPE t[][NUM])
{

	double alpha = 1.0, beta = 0.;
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NUM,NUM,NUM,alpha,(const double *)b,NUM,(const double *)a,NUM,beta,(double *)c,NUM);
}
