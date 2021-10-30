/*******************************************************************************
! Copyright(C) 2014-2015 Intel Corporation. All Rights Reserved.
!
! The source code, information  and  material ("Material") contained herein is
! owned  by Intel Corporation or its suppliers or licensors, and title to such
! Material remains  with Intel Corporation  or its suppliers or licensors. The
! Material  contains proprietary information  of  Intel or  its  suppliers and
! licensors. The  Material is protected by worldwide copyright laws and treaty
! provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
! modified, published, uploaded, posted, transmitted, distributed or disclosed
! in any way  without Intel's  prior  express written  permission. No  license
! under  any patent, copyright  or  other intellectual property rights  in the
! Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
! implication, inducement,  estoppel or  otherwise.  Any  license  under  such
! intellectual  property  rights must  be express  and  approved  by  Intel in
! writing.
! 
! *Third Party trademarks are the property of their respective owners.
! 
! Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
! this  notice or  any other notice embedded  in Materials by Intel or Intel's
! suppliers or licensors in any way.
!
!*******************************************************************************
!  Content:
!      Black-Scholes formula Example, main() module
!******************************************************************************/

#include <stdio.h>
#include "euro_opt.h"

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

double get_time_sec(){
	static struct timeval 	tv0;
	double time_, mytime;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	mytime = time_/1000000;
	return(mytime);
}

int main(int argc, char * argv[])
{
    int nopt = 4*1024*1024;
    tfloat *s0, *x, *t, *vcall_mkl, *vput_mkl, *vcall_compiler, *vput_compiler;
    tfloat vputs, vcalls;
    int i;

    /* Read nopt number of options parameter from command line */
    if (argc < 2)
    {
        printf("Usage: expect nopt input integer parameter, defaulting to %d\n", nopt);
    }
    else
    {
        sscanf(argv[1], "%d", &nopt);
    }

    /* Allocate arrays, generate input data */
    InitData( nopt, &s0, &x, &t, &vcall_compiler, &vput_compiler, &vcall_mkl, &vput_mkl );

    /* Compute call and put prices using compiler math libraries */
    double t0=get_time_sec();
    BlackScholesFormula_Compiler( nopt, RISK_FREE, VOLATILITY, s0, x, t, vcall_compiler, vput_compiler );
    double t1=get_time_sec();

    /* Compute call and put prices using MKL VML functions */
    BlackScholesFormula_MKL( nopt, RISK_FREE, VOLATILITY, s0, x, t, vcall_mkl, vput_mkl );
    double t2=get_time_sec();

    /* Display a few computed values */
    vcalls=vputs=0.0f;
    for (i=0; i<nopt; i++){
        vcalls+=vcall_compiler[i]/nopt;
        vputs+=vput_compiler[i]/nopt;
    }
    printf("call_compiler[0:%d]= %g\n", nopt, (double)(vcalls) );
    printf("put_compiler[0:%d]= %g\n", nopt, (double)(vputs) );
    printf("Time BlackScholesFormula_Compiler: %f sec.\n", t1-t0);
    printf("\n");
    vcalls=vputs=0.0f;
    for (i=0; i<nopt; i++){
        vcalls+=vcall_mkl[i]/nopt;
        vputs+=vput_mkl[i]/nopt;
    }
    printf("call_mkl[0:%d]= %g\n", nopt, (double)(vcalls) );
    printf("put_mkl[0:%d]= %g\n", nopt, (double)(vputs) );
    printf("Time BlackScholesFormula_MKL: %f sec.\n", t2-t1);

    /* Deallocate arrays */
    FreeData( s0, x, t, vcall_compiler, vput_compiler, vcall_mkl, vput_mkl );

    return 0;
}
