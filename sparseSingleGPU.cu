#include "sparseSingleGPU.cuh"

#include <algorithm>


sparseSingleGPU::sparseSingleGPU(const mxArray *sparseDouble) 
{
    
    if (!mxIsSparse(sparseDouble))
    {
        mexErrMsgIdAndTxt("MATLAB:sparseInternalOutput:invalidInputType",
                          "First argument must be sparse.");
    }

    mwIndex *ir, *jc; // ir: row indec, jc: encode row index and values in pr per coloumn
    double *pr; //value pointer
    
    // Get the starting pointer of all three data arrays.
    pr = mxGetPr(sparseDouble);     // row index array
    ir = mxGetIr(sparseDouble);     // row index array
    jc = mxGetJc(sparseDouble);     // column encrypt array
    mwSize nCols = mxGetN(sparseDouble);       // number of columns
    mwSize nRows = mxGetM(sparseDouble);       // number of rows

    // nnz = mxGetNzmax(prhs[0]); // number of possible non zero elements
    mwSize nnz = jc[nCols]; // number of non zero elements currently stored inside the sparse matrix

    #ifdef NDEBUG
        mexPrintf("Creating Sparse matrix of size %dx%d (nnz=%d)\n",nRows,nCols,nnz);
    #endif

    val_t* pr_d;
    index_t* ir_d;
    index_t* jc_d;
    
    //Create the CUDA Sparse Matrix        
    try {        
        //Create the sparse arrays on the GPU
        
        /*
        const mwSize ndims = 1; 
        //For now, use mxGPUArrays. We will only need this finally if we want to expose these arrays to Matlab. Alternatively, one could also expose them only on demand?
        mxGPUArray* d_pr = mxGPUCreateGPUArray(ndims, nnz, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        mxGPUArray* d_ir = mxGPUCreateGPUArray(ndims, nnz, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        mxGPUArray* d_jc = mxGPUCreateGPUArray(ndims, nnz, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        */

        cudaEvent_t event_start, event_start2, event_copyVal, event_copyIx, event_copyCols, event_createHandle, event_createMat;

        CHECK_CUDA(cudaEventCreate(&event_start));
        CHECK_CUDA(cudaEventCreate(&event_start2));
        CHECK_CUDA(cudaEventCreate(&event_copyVal));
        CHECK_CUDA(cudaEventCreate(&event_copyIx));
        CHECK_CUDA(cudaEventCreate(&event_copyCols));
        CHECK_CUDA(cudaEventCreate(&event_createHandle));
        CHECK_CUDA(cudaEventCreate(&event_createMat));

        CHECK_CUDA(cudaEventRecord(event_start,0));

        cudaStream_t stream1, stream2, stream3;
        CHECK_CUDA(cudaStreamCreate(&stream1));
        CHECK_CUDA(cudaStreamCreate(&stream2));
        CHECK_CUDA(cudaStreamCreate(&stream3));
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                val_t* castedHost = this->cudaMallocAndMemcpyToDeviceWithCast<val_t,double>(&pr_d, pr, nnz, stream1);
                CHECK_CUDA(cudaEventRecord(event_copyVal,stream1));
                CHECK_CUDA(cudaStreamSynchronize(stream1));
                if (castedHost)
                    delete castedHost;
            }

            #pragma omp section
            {
                index_t* castedHost = this->cudaMallocAndMemcpyToDeviceWithCast<index_t,mwIndex>(&ir_d, ir, nnz, stream2);
                CHECK_CUDA(cudaEventRecord(event_copyIx,stream2));
                CHECK_CUDA(cudaStreamSynchronize(stream2));
                if (castedHost)
                    delete castedHost;
            }

            #pragma omp section
            {
                index_t* castedHost = this->cudaMallocAndMemcpyToDeviceWithCast<index_t,mwIndex>(&jc_d, jc, mwSize(nCols+1), stream3);
                CHECK_CUDA(cudaEventRecord(event_copyCols,stream3));
                CHECK_CUDA(cudaStreamSynchronize(stream3));
                if (castedHost)
                    delete castedHost;
            }       
        }
        
        cusparseHandle_t cuSparseHandle;
        
        CHECK_CUDA(cudaEventRecord(event_start2,0));  
        CHECK_CUSPARSE(cusparseCreate(&cuSparseHandle));
        CHECK_CUDA(cudaEventRecord(event_createHandle,0));       
        

        #ifdef NDEBUG
            mexPrintf("Device Pointers:\n\t%d\n\t%d\n\t%d\n",pr_d,ir_d,jc_d);
        #endif

        //this->cudaSpMatrix = std::make_shared<cusparseSpMatDescr_t>();
        
        CHECK_CUDA(cudaDeviceSynchronize());
        
        cusparseSpMatDescr_t cuSparseMatrix;        

        //TODO match CUDA types to index_t and val_t
        //the cusparseCreateCsc function only became available later in CUDA toolkit 11. We can though initialize a CSR matrix with similar storage pattern and say our matrix is transposed        
        #ifdef CUDA_SPMAT_CSR
            CHECK_CUSPARSE(cusparseCreateCsr(&cuSparseMatrix, nCols, nRows, nnz, (void*) jc_d, (void*) ir_d, (void*) pr_d, cusparseType<index_t>::kind, cusparseType<index_t>::kind, CUSPARSE_INDEX_BASE_ZERO, cudaType<val_t>::kind));
        #else
            CHECK_CUSPARSE(cusparseCreateCsc(&cuSparseMatrix, nRows, nCols, nnz, (void*) jc_d, (void*) ir_d, (void*) pr_d, cusparseType<index_t>::kind, cusparseType<index_t>::kind, CUSPARSE_INDEX_BASE_ZERO, cudaType<val_t>::kind));
        #endif

        CHECK_CUDA(cudaEventRecord(event_createMat,0));
        CHECK_CUDA(cudaEventSynchronize(event_createMat));

        this->cudaSpMatrix = std::make_shared<sparseSingleGPUdata>(pr_d, ir_d, jc_d, nRows, nCols, nnz, cuSparseHandle, cuSparseMatrix);        

        float time_copyVal, time_copyIx, time_copyCols, time_createHandle, time_createMat, timeAll;
        CHECK_CUDA(cudaEventElapsedTime(&time_copyVal,event_start,event_copyVal));
        CHECK_CUDA(cudaEventElapsedTime(&time_copyIx,event_start,event_copyIx));
        CHECK_CUDA(cudaEventElapsedTime(&time_copyCols,event_start,event_copyCols));
        CHECK_CUDA(cudaEventElapsedTime(&time_createHandle,event_start2,event_createHandle));
        CHECK_CUDA(cudaEventElapsedTime(&time_createMat,event_createHandle,event_createMat));
        CHECK_CUDA(cudaEventElapsedTime(&timeAll,event_start,event_createMat));

        mexPrintf("GPU Times Construct (%3.1f ms):\n\tcopy values (async): %3.1f ms\n\tcopy indices (async): %3.1f ms\n\tcopy col starts (async): %3.1f ms\n\tcusparse Initialization: %3.1f ms\n\tmatrix creation: %3.1f ms\n",
            timeAll,time_copyVal,time_copyIx,time_copyCols,time_createHandle,time_createMat);

    }
    catch (...) {
        mexErrMsgIdAndTxt("MATLAB:sparseInternalOutput:invalidInputType","CUDA sparse matrix could not be constructed!");
    }

    // no need to free memory because matlab should handle memory management of return values
}

sparseSingleGPU::~sparseSingleGPU()
{
    #ifdef NDEBUG 
        mexPrintf("Calling destructor! We still have %d references to the data object!\n",this->cudaSpMatrix.use_count() - 1);
    #endif
}

//// Getters & Setters ////
mwSize sparseSingleGPU::getNnz() const {
    return this->cudaSpMatrix->nnz;
}

mwSize sparseSingleGPU::getCols() const {
    if (this->transposed)
        return this->cudaSpMatrix->nRows;
    else
        return this->cudaSpMatrix->nCols;
}

mwSize sparseSingleGPU::getRows() const {
    if (this->transposed)
        return this->cudaSpMatrix->nCols;
    else
        return this->cudaSpMatrix->nRows;
}

mxArray* sparseSingleGPU::timesVec(const mxSingle* vals,mwSize n, bool transposeInPlace) const 
{
    //CUDA performance
    cudaEvent_t event1_start, event2_prepared, event3_spmv, event4_result, event5_cleanup;
    CHECK_CUDA(cudaEventCreate(&event1_start));
    CHECK_CUDA(cudaEventCreate(&event2_prepared));
    CHECK_CUDA(cudaEventCreate(&event3_spmv));
    CHECK_CUDA(cudaEventCreate(&event4_result));
    CHECK_CUDA(cudaEventCreate(&event5_cleanup));

    //Memory allocation & copy
    CHECK_CUDA(cudaEventRecord(event1_start,0));

    cudaStream_t stream1, stream2, stream3;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUDA(cudaStreamCreate(&stream3));

    //Allocate & copy multiplication vector on stream 1
    val_t* vals_d = nullptr;
    val_t* vals_h_casted = this->cudaMallocAndMemcpyToDeviceWithCast<val_t,const mxSingle>(&vals_d,vals,n,stream1);    

    //Prepare and allocate result vector on stream 2
    mwSize m;
    if (transposeInPlace)
        m = this->getCols();
    else
        m = this->getRows();
    
    mxArray* result = mxCreateNumericMatrix(m,1,mxSINGLE_CLASS,mxREAL);
    mxSingle* result_data = mxGetSingles(result);

    val_t* result_d = nullptr;
    CHECK_CUDA(cudaMallocAsync((void**)&result_d, m * sizeof(val_t),stream2));
    
    // CUSPARSE SpMV: Y=α*op(A)⋅X+β*Y
    //Prepare data structure on host and stream 3

    //XOR to evaluate if we should transpose
    transposeInPlace = (transposeInPlace != this->transposed);

    #ifdef CUDA_SPMAT_CSR
        //note that the op (transpose) is the other way round due to storing the matrix as transposed CSR on the device
        cusparseOperation_t op_transpose = transposeInPlace ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE; 
    #else
        cusparseOperation_t op_transpose = transposeInPlace ? CUSPARSE_OPERATION_TRANSPOSE: CUSPARSE_OPERATION_NON_TRANSPOSE; 
    #endif

    cusparseDnVecDescr_t vecX, vecY;
    void* d_buffer = NULL;
    size_t bufferSize = 0;
    val_t alpha = 1.0f;
    val_t beta = 0.0f;

    cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_ALG_DEFAULT; //Corresponds to CUSPARSE_SPMV_CSR_ALG1
    //cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_CSR_ALG1; //Not deterministic for similar input (fastest)
    //cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_CSR_ALG2; //Deterministic version for similar input (slower)

    //CHECK_CUSPARSE(cusparseCreate(&cuSparseHandle)); //Now done in the constructor

    CHECK_CUSPARSE(cusparseSetStream(this->cudaSpMatrix->cuSparseHandle,stream3));

    // create dense vector x for op(A)*x
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, vals_d, cudaType<val_t>::kind) );

    // create dense output vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, m, result_d, cudaType<val_t>::kind));

    // create workspace buffer if needed
    CHECK_CUSPARSE(
        cusparseSpMV_bufferSize(
            this->cudaSpMatrix->cuSparseHandle, op_transpose,
            &alpha, this->cudaSpMatrix->cuSparseMatrix, vecX, &beta, vecY,cudaType<val_t>::kind,
            algorithm, &bufferSize));
    if (bufferSize > 0)
        CHECK_CUDA(cudaMallocAsync(&d_buffer, bufferSize,stream3));     
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(event2_prepared,0));
    CHECK_CUSPARSE(cusparseSetStream(this->cudaSpMatrix->cuSparseHandle,0));
    
    // execute SpMV
    //CHECK_CUSPARSE(cusparseSpMV(handle, trans, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
    CHECK_CUSPARSE(cusparseSpMV(this->cudaSpMatrix->cuSparseHandle, op_transpose, &alpha, this->cudaSpMatrix->cuSparseMatrix, vecX, &beta, vecY, cudaType<val_t>::kind, algorithm, d_buffer));

    CHECK_CUDA(cudaEventRecord(event3_spmv,0));

    //Copy result
    CHECK_CUDA(cudaMemcpy(result_data, result_d, m * sizeof(mxSingle), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(event4_result,0));

    //Synchronize - should be done with memcpy anywas
    //CHECK_CUDA(cudaDeviceSynchronize()); 
    
    //Free buffers etc.
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    //CHECK_CUSPARSE(cusparseDestroy(cuSparseHandle)); //NOw done in destructor
    if (d_buffer) CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaFree(result_d));
    CHECK_CUDA(cudaFree(vals_d));

    CHECK_CUDA(cudaEventRecord(event5_cleanup,0));
    CHECK_CUDA(cudaEventSynchronize(event5_cleanup));

    float time2_prepared, time3_spmv, time4_result, time5_cleanup, timeAll;
    CHECK_CUDA(cudaEventElapsedTime(&time2_prepared,event1_start,event2_prepared));
    CHECK_CUDA(cudaEventElapsedTime(&time3_spmv,event2_prepared,event3_spmv));
    CHECK_CUDA(cudaEventElapsedTime(&time4_result,event3_spmv,event4_result));
    CHECK_CUDA(cudaEventElapsedTime(&time5_cleanup,event4_result,event5_cleanup));
    CHECK_CUDA(cudaEventElapsedTime(&timeAll,event1_start,event5_cleanup));

    #ifdef CUDA_MEX_PERFANA
        mexPrintf("GPU Times (All=%3.1f ms):\n\tprepare: %3.1f ms\n\tSpMv: %3.1f ms\n\tcopy result: %3.1f ms\n\tcleanup: %3.1f ms\n",
            timeAll,time2_prepared, time3_spmv, time4_result, time5_cleanup);
    #endif

    //return the bare array
    return result;
}

sparseSingleGPU::sparseSingleGPU(std::shared_ptr<sparseSingleGPUdata> cudaSpMatrix_) {
    this->cudaSpMatrix = cudaSpMatrix_;
}

sparseSingleGPU* sparseSingleGPU::transpose() const {
    sparseSingleGPU* transposedCopy = new sparseSingleGPU(this->cudaSpMatrix);
    transposedCopy->transposed = !this->transposed;
    
    return transposedCopy;
}

mxArray* sparseSingleGPU::vecTimes(const mxSingle* vals,mwSize n, bool transposeInPlace) const 
{
    //Result Dimension: We expect a row vector from x*A (x is a row vector)
    mwSize resultDim[2];
    resultDim[0] = 1;
    resultDim[1] = this->getCols();
    
    //Call A*x with transposed matrix. We use try catch to be able to perform the transpose reverse even when the operation fails
    mxArray* result = this->timesVec(vals,n,true);
    
    //Transpose result efficiently
    int status = mxSetDimensions(result,resultDim,2);      

    return result;
}