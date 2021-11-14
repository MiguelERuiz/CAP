# ==============================================================
# 
# SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
# 
# Copyright 2018 Intel Corporation
# 
# THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
# 
# =============================================================
#
# To compile with the Intel(R) C++ Compiler for Linux*, creating an execution
# file with the extension ".icc":
#
#   Source <path_to_compiler_bin>/compilervars.(sh|csh) or iccvars.(sh|csh);
#
#   > make icc
#



SHELL = /bin/sh


PARAMODEL = -DUSE_OMP -fopenmp	# Use OpenMP for multithreading

ICC     = icc
CFLAGS  = -g -O3
OPTFLAGS = -xhost 


ICFLAGS = $(CFLAGS) $(PARAMODEL)-DICC -debug inline-debug-info #-vec-report3 -qopt-report -qopt-report-phase=vec
LDFLAGS = -mkl

ICC_EXE = matrix.icc


srcdir = src

icc: $(ICC_EXE)

OBJS = util.o thrmodel.o multiply.o matrix.o 


matrix.icc: $(srcdir)/matrix.c $(srcdir)/multiply.c $(srcdir)/multiply.h $(srcdir)/util.c $(srcdir)/thrmodel.c
	$(ICC) $(ICFLAGS) -c $(srcdir)/util.c -D_LINUX
	$(ICC) $(ICFLAGS) -c $(srcdir)/thrmodel.c -D_LINUX
	$(ICC) $(ICFLAGS) $(OPTFLAGS) -c $(srcdir)/multiply.c -D_LINUX
	$(ICC) $(ICFLAGS) $(OPTFLAGS) -c $(srcdir)/matrix.c -D_LINUX
	$(ICC) $(ICFLAGS) $(OBJS) -o $@ $(LDFLAGS)



clean:
	@rm -rf $(OBJS) $(ICC_EXE) *.o 

# * Other names and brands may be claimed as the property of others.
