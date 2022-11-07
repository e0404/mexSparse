#include "mex.h"
#include "C___class_interface/class_handle.hpp"
#include "sparseSingle.hpp"

#include <stdexcept>

// The class that we are interfacing to


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
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
        if (nrhs == 1)
            plhs[0] = convertPtr2Mat<sparseSingle>(new sparseSingle());
        else if (nrhs == 2) {
            if (!mxIsSparse(prhs[1]))
            {
                mexErrMsgIdAndTxt("MATLAB:sparseInternalOutput:invalidInputType",
                                  "single sparse matrix can only be constructed from double sparse matrix.");
            }
            try
            {
                plhs[0] = convertPtr2Mat<sparseSingle>(new sparseSingle(prhs[1]));
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
    sparseSingle* sparseSingle_instance = convertMat2Ptr<sparseSingle>(prhs[1]);
    
    if (!strcmp("nnz",cmd))
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("nnz: Unexpected arguments.");
        mwSize nnz = sparseSingle_instance->getNnz();
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
            pr[0] = static_cast<double>(sparseSingle_instance->getRows());
            pr[1] = static_cast<double>(sparseSingle_instance->getCols());
            plhs[0] = szArray;
        }
        catch (...)
        {
            mexErrMsgTxt("size: Unexpected access violation.");
        }

        return;
    }

    if (!strcmp("disp",cmd))
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("full: Unexpected arguments.");
        try {
            sparseSingle_instance->disp();        
        }
        catch (...)
        {
            mexErrMsgTxt("disp: Unexpected access violation.");
        }
        return;
    }

    if (!strcmp("full",cmd))
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("full: Unexpected arguments.");
        try {
            mxArray* result = sparseSingle_instance->full();
            plhs[0] = result;       
        }
        catch (...)
        {
            mexErrMsgTxt("full: Unexpected access violation.");
        }
        return;
    }

    if (!strcmp("timesVec",cmd))
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("timesVec: Unexpected arguments.");
        try {
            const mxSingle* vals = mxGetSingles(prhs[2]);
            sparseSingle::index_t n = mxGetNumberOfElements(prhs[2]);
            mxArray* result = sparseSingle_instance->timesVec(vals,n);
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
            sparseSingle::index_t n = mxGetNumberOfElements(prhs[2]);
            mxArray* result = sparseSingle_instance->vecTimes(vals,n);
            plhs[0] = result;
        }
        catch(...)
        {
            mexErrMsgTxt("timesVec: Product failed.");
        }
        return;
    }

    if (!strcmp("timesScalar",cmd))
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgTxt("vecTimes: Unexpected arguments.");
        try {
            if (mxIsScalar(prhs[2]))
                mexErrMsgTxt("timesScalar: Unexpected arguments.");
            const mxSingle* vals = mxGetSingles(prhs[2]);
            mxSingle scalar = vals[0];
            sparseSingle* result = sparseSingle_instance->timesScalar(scalar);
            plhs[0] = convertPtr2Mat<sparseSingle>(result);
        }
        catch(...)
        {
            mexErrMsgTxt("timesVec: Product failed.");
        }
        return;
    }

    if (!strcmp("transpose",cmd))
    {
        if (nlhs < 0 || nlhs > 1 || nrhs > 2)
            mexErrMsgTxt("transpose: Unexpected arguments.");
        try {
            plhs[0] = convertPtr2Mat<sparseSingle>(sparseSingle_instance->transpose());
        }
        catch(...)
        {
            mexErrMsgTxt("transpose: Transposing failed.");
        }
        return;
    }

    if (!strcmp("subsrefRowCol",cmd))
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 4)
            mexErrMsgTxt("subsrefRowCol: Unexpected arguments.");
        try {
            //mxArray* result = sparseSingle_instance->linearIndexing(prhs[2]);
            
            sparseSingle* result = sparseSingle_instance->rowColIndexing(prhs[2],prhs[3]);
            plhs[0] = convertPtr2Mat<sparseSingle>(result);
        }
        catch(...)
        {
            mexErrMsgTxt("subsrefRowCol: Unexpected arguments.");
        }
        return;
    }

    
    if (!strcmp("linearIndexing",cmd))
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgTxt("linear indexing: Unexpected arguments.");

        try {
            sparseSingle* result = sparseSingle_instance->linearIndexing(prhs[2]);
            //result = sparseSingle_instance->allValues()
            plhs[0] = convertPtr2Mat<sparseSingle>(result);
        }
        catch(std::exception &e)
        {
            mexErrMsgIdAndTxt("mexSparseSingle:stl_error","Exception from Stl: %s",e.what());
        }
        catch(...)
        {
            mexErrMsgTxt("linear Indexing failed.");
        }
        return;
    }
    


    // Delete
    if (!strcmp("delete", cmd)) {
        // Destroy the C++ object
        destroyObject<sparseSingle>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        return;
    }
    
    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized.");
}
