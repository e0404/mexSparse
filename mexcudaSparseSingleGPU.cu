/**************************************************************************
*
* Copyright 2022 the matRad development team.
*
**************************************************************************/

/*
Mex Function for Computing a sparse vector product with
compiling needs a matlab supported c/c++ compiler e.g. Microsoft Visual Studio C++ or MinGW64 and CUDA
compile with matlab: mexcuda  'mexcudaSparseSingleGPU.cu' ...
        -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\include" ...
        -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\lib\x64" ...
        NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"'...
        -lcusparse -dynamic -v 
look into compileAll and compileCUDA for more informations
run with matlab: ret_v = SparseSingleGPU(nrows, ncols, nnz, jc, ir, pr, trans, vector);
*look into SparseSingleGP for more informations
*/

// include matlabs api
#include "mex.h"
#include "class_handle.hpp"
//#include "gpu/mxGPUArray.h"
#include "matrix.h"

#include "sparseSingleGPU.cuh"

void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[])
{
    //int initalizeMxGPU = mxInitGPU();

    //if (initalizeMxGPU == MX_GPU_FAILURE)
    //    mexErrMsgTxt("Could not initialize GPU!");

    // Get the command string
    char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");
        
    // New
    if (!strcmp("new", cmd)) {
        // Check parameters
        if (nlhs != 1)
            mexErrMsgTxt("New: One output expected.");
        // Return a handle to a new C++ instance
        //if (nrhs == 1)
        //    plhs[0] = convertPtr2Mat<sparseSingleGPU>(new sparseSingleGPU());
        //else if (nrhs == 2) {
        if (nrhs == 2) {
            if (!mxIsSparse(prhs[1]))
            {
                mexErrMsgIdAndTxt("MATLAB:sparseInternalOutput:invalidInputType",
                                  "single sparse matrix can only be constructed from double sparse matrix.");
            }
            try
            {
                plhs[0] = convertPtr2Mat<sparseSingleGPU>(new sparseSingleGPU(prhs[1]));
            }
            catch (...)
            {
                mexErrMsgIdAndTxt("MATLAB:sparseInternalOutput:invalidInputType", "single sparse matrix could not be constructed from double.");
            }
        }
        else
            mexErrMsgTxt("New: Invalid Input.");
        // We return now, as the object is constructed
        return;
    }    
    
    // For all other purposes, we need to pass the class handle as second argument.
    //Check there is a second input, which should be the class instance handle
    if (nrhs < 2)
		mexErrMsgTxt("Second input should be a class instance handle.");
    
    // Get the class instance pointer from the second input
    sparseSingleGPU* sparseSingleGPU_instance = convertMat2Ptr<sparseSingleGPU>(prhs[1]);
    
    if (!strcmp("nnz",cmd))
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("nnz: Unexpected arguments.");
        mwSize nnz = sparseSingleGPU_instance->getNnz();
        plhs[0] = mxCreateDoubleScalar((double) nnz);
        return;
    }

    if (!strcmp("size",cmd))
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("size: Unexpected arguments.");
        try {
            mxArray* szArray = mxCreateDoubleMatrix(1,2,mxREAL);
            double* pr = mxGetPr(szArray);
            pr[0] = static_cast<double>(sparseSingleGPU_instance->getRows());
            pr[1] = static_cast<double>(sparseSingleGPU_instance->getCols());
            plhs[0] = szArray;
        }
        catch (...)
        {
            mexErrMsgTxt("size: Unexpected access violation.");
        }

        return;
    }

    /*

    if (!strcmp("disp",cmd))
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("size: Unexpected arguments.");
        try {
            sparseSingleGPU_instance->disp();        
        }
        catch (...)
        {
            mexErrMsgTxt("disp: Unexpected access violation.");
        }
        return;
    }
    */

    if (!strcmp("transpose",cmd))
    {
        if (nlhs < 0 || nlhs > 1 || nrhs > 2)
            mexErrMsgTxt("transpose: Unexpected arguments.");
        try {
            plhs[0] = convertPtr2Mat<sparseSingleGPU>(sparseSingleGPU_instance->transpose());
        }
        catch(...)
        {
            mexErrMsgTxt("transpose: Transposing failed.");
        }
        return;
    }

    if (!strcmp("timesVec",cmd))
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("timesVec: Unexpected arguments.");
        try {
            const mxSingle* vals = mxGetSingles(prhs[2]);
            sparseSingleGPU::index_t n = mxGetNumberOfElements(prhs[2]);
            mxArray* result = sparseSingleGPU_instance->timesVec(vals,n);
            plhs[0] = result;
        }
        catch(...)
        {
            mexErrMsgTxt("timesVec: Product failed.");
        }
        return;
    }

    if (!strcmp("vecTimes",cmd))
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("vecTimes: Unexpected arguments.");
        try {
            const mxSingle* vals = mxGetSingles(prhs[2]);
            sparseSingleGPU::index_t n = mxGetNumberOfElements(prhs[2]);
            mxArray* result = sparseSingleGPU_instance->vecTimes(vals,n);
            plhs[0] = result;
        }
        catch(...)
        {
            mexErrMsgTxt("timesVec: Product failed.");
        }
        return;
    }  

    // Delete
    if (!strcmp("delete", cmd)) {
        // Destroy the C++ object
        destroyObject<sparseSingleGPU>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        return;
    }

    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized.");
}