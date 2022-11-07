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
#include "C___class_interface/class_handle.hpp"
//#include "gpu/mxGPUArray.h"
#include "matrix.h"

#include "sparseSingleGPU.cuh"

//define input arguments for less confusion
/*
#define NROWS_A prhs[0]
#define NCOLS_A prhs[1]
#define NNZ_A prhs[2]
#define JC_A prhs[3] // column offset size cols + 1
#define IR_A prhs[4] // row index size nnz
#define PR_A prhs[5] // values size nnz
#define TRANS prhs[6] // transpose flag
#define X_B prhs[7] // input Vector
*/


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

    /*
    char const *const errId = "matRad:gpuAcceleration:cuSparse:InvalidInput";
    char const *const errMsg = "Invalid input to MEX file";
     
    //check input and output arguments
    
    if (nrhs != 8) mexErrMsgIdAndTxt(errId, "Wrong number of input arguments");

    if (!mxIsScalar(NROWS_A)) mexErrMsgIdAndTxt(errId, "Argument 1 ROWS must be scalar");
    if (!mxIsScalar(NCOLS_A)) mexErrMsgIdAndTxt(errId, "Argument 2 COLS must be scalar");
    if (!mxIsScalar(NNZ_A)) mexErrMsgIdAndTxt(errId, "Argument 3 NNZ must be scalar");
    if (!mxIsScalar(TRANS)) mexErrMsgIdAndTxt(errId, "Argument 7 Transpose Flag must be scalar");

    if (!mxIsGPUArray(JC_A) && !mxGPUIsValidGPUData(JC_A)) mexErrMsgIdAndTxt(errId, "Argument 4 JC must be gpu array");
    if (!mxIsGPUArray(IR_A) && !mxGPUIsValidGPUData(IR_A)) mexErrMsgIdAndTxt(errId, "Argument 5 IR must be gpu array");
    if (!mxIsGPUArray(PR_A) && !mxGPUIsValidGPUData(PR_A)) mexErrMsgIdAndTxt(errId, "Argument 6 PR must be gpu array");
    if (!mxIsGPUArray(X_B) && !mxGPUIsValidGPUData(X_B)) mexErrMsgIdAndTxt(errId, "Argument 7 Vector B must be gpu array");

    // Initializie MathWorks Parallel Gpu API
    mxInitGPU();

    // Create read only pointer to gpu arrays
    mxGPUArray const *ir_a = mxGPUCreateFromMxArray(IR_A);
    mxGPUArray const *jc_a = mxGPUCreateFromMxArray(JC_A);
    mxGPUArray const *pr_a = mxGPUCreateFromMxArray(PR_A);

    mxGPUArray const *x = mxGPUCreateFromMxArray(X_B);

    mwSize A_n_rows = mxGetScalar(NROWS_A);
    mwSize A_n_cols = mxGetScalar(NCOLS_A);
    mwSize A_nnz = mxGetScalar(NNZ_A);

    mwSize *xdims = (mwSize*)mxGPUGetDimensions(x);

    if (mxGPUGetNumberOfDimensions(x) > 2) mexErrMsgIdAndTxt(errId, "Vector has to many dimensions");

    mwSize numelx = mxGPUGetNumberOfElements(x);
    cusparseOperation_t trans = (cusparseOperation_t)mxGetScalar(TRANS);
    cusparseOperation_t trans_csr = (cusparseOperation_t) !mxGetScalar(TRANS);
    //int nx = (trans == CUSPARSE_OPERATION_NON_TRANSPOSE) ? xdims[0] : xdims[1];

    // check if size allows multiplication
    //mexPrintf("vector Dimensions x:%d y:%d \n", xdims[0], xdims[1]);
    //mexPrintf("numel in vector: %d\n, number of dimensions in vector: %d\n", mxGPUGetNumberOfElements(x), mxGPUGetNumberOfDimensions(x));
    //mexPrintf("A number cols: %d number row:%d \n", A_n_cols, A_n_rows);
    if (trans == CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
        if (numelx != A_n_cols)
            mexErrMsgIdAndTxt(errId, "Vector argument wrong size for multiply");
    }
    else
    {
        if (numelx != A_n_rows)
            mexErrMsgIdAndTxt(errId, "Vector argument wrong size for transpose multiply");
    }

    // check types
    if (mxGPUGetClassID(ir_a) != mxINT32_CLASS) mexErrMsgIdAndTxt(errId, "IR is not int32");
    if (mxGPUGetClassID(jc_a) != mxINT32_CLASS) mexErrMsgIdAndTxt(errId, "JC is not int32");
    if (mxGPUGetClassID(pr_a) != mxSINGLE_CLASS) mexErrMsgIdAndTxt(errId, "VAL is not single");
    if (mxGPUGetClassID(x) != mxSINGLE_CLASS) mexErrMsgIdAndTxt(errId, "Vector V is not single");

    // check complexity
    if (mxGPUGetComplexity(pr_a) != mxREAL) mexErrMsgIdAndTxt(errId, "Complex arguments are not supported");
    if (mxGPUGetComplexity(x) != mxREAL) mexErrMsgIdAndTxt(errId, "Complex arguments are not supported");


    // return vector
    const mwSize ndim = 1;
    mwSize dims[ndim] = { trans == CUSPARSE_OPERATION_NON_TRANSPOSE ? A_n_rows : A_n_cols };
    mxClassID cid = mxGPUGetClassID(x);
    mxGPUArray* y;

    y = mxGPUCreateGPUArray(ndim, dims, cid, mxREAL, MX_GPU_INITIALIZE_VALUES);
    if (y == NULL) mexErrMsgIdAndTxt(errId, "mxGPUCreateGPUArray failed");

    // CUSPARSE APIs Y=α*op(A)⋅X+β*Y
    cusparseHandle_t handle = NULL;
    cusparseStatus_t status;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* d_buffer = NULL;
    size_t bufferSize = 0;

    CHECK_CUSPARSE( cusparseCreate(&handle) );

    // Convert matlab pointer to native pointer and types
    int* const d_ir_a = (int*)mxGPUGetDataReadOnly(ir_a); // data row index of a
    int* const d_jc_a = (int*)mxGPUGetDataReadOnly(jc_a); // data coloumn indexing of a
    float * const d_val = (float *)mxGPUGetDataReadOnly(pr_a); // data values of a
    float * const d_x = (float *)mxGPUGetDataReadOnly(x); // data in vector x
    float* d_y = (float *)mxGPUGetData(y); // data in (return) vector y
    float alpha = 1.0f;
    float beta = 0.0f;

    // create sparse matrix A
    //CHECK_CUSPARSE( cusparseCreateCsc(&matA, A_n_rows, A_n_cols, A_nnz, d_jc_a, d_ir_a, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_n_cols, A_n_rows, A_nnz, d_jc_a, d_ir_a, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

    // create dense vector x
    int x_numel = (trans == CUSPARSE_OPERATION_NON_TRANSPOSE) ? A_n_cols: A_n_rows;
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, x_numel, d_x, CUDA_R_32F) );

    // create dense output vector y
    int y_numel = (trans == CUSPARSE_OPERATION_NON_TRANSPOSE) ? A_n_rows : A_n_cols;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, y_numel, d_y, CUDA_R_32F));
    
    
    // create buffer if needed
    // CHECK_CUSPARSE(
    //     cusparseSpMV_bufferSize(
    //         handle, trans,
    //         &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
    //         CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    
    
    CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(
            handle, trans_csr,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    if (bufferSize > 0)
    {
        cudaError_t status = cudaMalloc(&d_buffer, bufferSize);
        if (status != cudaSuccess)
            mexErrMsgIdAndTxt(errId, "Critical CUSPARSE ERROR");
    }

    // execute SpMV
    //CHECK_CUSPARSE(cusparseSpMV(handle, trans, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
    CHECK_CUSPARSE(cusparseSpMV(handle, trans_csr, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

    //return result this status check has some problems and return unrecognized error codes now and then propable when no status was set beforehand or another gpu operation writes into status in between operations
    // if (status == CUSPARSE_STATUS_SUCCESS)
    // {
    //     plhs[0] = mxGPUCreateMxArrayOnGPU(y);
    // }else
    // {
    //     mexPrintf("CUDA failed at %d line with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status);
    //     mexErrMsgTxt("Unkown Error in cu sparse");
    // }

    // return data
    CHECK_CUDA(cudaDeviceSynchronize()); // GPUMex should handle the sychronization, but its not clear from the documentation
    plhs[0] = mxGPUCreateMxArrayOnGPU(y);

    // free data
    // destroy cuda matrix/ vector descriptors and buffer
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    if (d_buffer) CHECK_CUDA(cudaFree(d_buffer));
    mxGPUDestroyGPUArray(ir_a);
    mxGPUDestroyGPUArray(pr_a);
    mxGPUDestroyGPUArray(jc_a);
    mxGPUDestroyGPUArray(x);
    mxGPUDestroyGPUArray(y);
    mxFree(xdims);

    return;
    */

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