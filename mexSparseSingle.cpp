#include "mex.h"
#include "mexSparseInterface.hpp"

#include "sparseEigen.hpp"

typedef sparseEigen<int64_t,mxSingle> sparseSingle;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mexSparseInterface<sparseSingle>(nlhs, plhs, nrhs, prhs);
}
