#include "mex.h"
#include "mexSparseInterface.hpp"

#include "sparseEigen.hpp"

typedef sparseEigen<int64_t,mxDouble> sparseDouble;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mexSparseInterface<sparseDouble>(nlhs, plhs, nrhs, prhs);
}
