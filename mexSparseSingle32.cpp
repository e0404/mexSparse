#include "mex.h"
#include "sparseEigen.hpp"
#include "mexSparseInterface.hpp"

#include <stdexcept>

typedef sparseEigen<int32_t,mxSingle> sparseSingle32;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mexSparseInterface<sparseSingle32>(nlhs, plhs, nrhs, prhs);
}
