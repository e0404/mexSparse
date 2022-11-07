#pragma once

#include "mex.h"
//#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include <memory>
#include <algorithm>
#include <execution>

// include Cuda runtime and Cusparse
#include <cuda.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>

//#include "sparseSingle.hpp"

#define CUDA_MEX_ERRID "mexSparseSingleCUDA::CriticalError"
//#define CUDA_MEX_PERFANA

//Use the CSR initialization instead of the CSC (due to availability of CSC initialization)
//#define CUDA_SPMAT_CSR

// checks for simplifying cuda code
#define CHECK_CUDA(func)                                       \
{                                                           \
    cudaError_t status = (func);                            \
    if (status != cudaSuccess)                              \
    {                                                       \
        mexPrintf("CUDA failed at %d line with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status); \
        mexErrMsgIdAndTxt(CUDA_MEX_ERRID, "Critical CUDA ERROR"); \
    }                                                       \
}

#define CHECK_CUSPARSE(func)               \
{                                          \
    cusparseStatus_t status = (func);      \
    if (status != CUSPARSE_STATUS_SUCCESS) \
    {                                      \
        mexPrintf("CUDA failed at %d line with error: %s (%d)\n", __LINE__, cusparseGetErrorString(status), status); \
        mexErrMsgIdAndTxt(CUDA_MEX_ERRID, "Critical CUSPARSE ERROR"); \
    }                                                       \
}

//CUDA Types
template<typename T> struct cudaType {};  
template<typename T> struct cusparseType {};  
template<> struct cudaType<float> {
    using type = float;
    static constexpr cudaDataType_t kind = CUDA_R_32F;
};
template<> struct cudaType<double> {
    using type = double;
    static constexpr cudaDataType_t kind = CUDA_R_64F;
};
template<> struct cusparseType<int32_t> {
    using type = int32_t;
    static constexpr cusparseIndexType_t kind = CUSPARSE_INDEX_32I;
};
template<> struct cusparseType<int64_t> {
    using type = int64_t;
    static constexpr cusparseIndexType_t kind = CUSPARSE_INDEX_64I;
};

class sparseSingleGPU
{
public:    

    typedef int32_t index_t; //most cuSparse operations only supported with int32_t
    typedef float val_t;

    struct sparseSingleGPUdata
    {
        sparseSingleGPU::val_t* pr_d;
        sparseSingleGPU::index_t* ir_d;
        sparseSingleGPU::index_t* jc_d;

        cusparseHandle_t cuSparseHandle;
        cusparseSpMatDescr_t cuSparseMatrix;

        mwSize nRows;
        mwSize nCols;
        mwSize nnz;

        sparseSingleGPUdata(sparseSingleGPU::val_t* pr_d_, 
                            sparseSingleGPU::index_t* ir_d_, 
                            sparseSingleGPU::index_t* jc_d_, 
                            mwSize nRows_, mwSize nCols_, mwSize nnz_,
                            cusparseHandle_t cuSparseHandle_, 
                            cusparseSpMatDescr_t cuSparseMatrix_) :
            pr_d(pr_d_), ir_d(ir_d_), jc_d(jc_d_), nRows(nRows_), nCols(nCols_), nnz(nnz_), cuSparseHandle(cuSparseHandle_), cuSparseMatrix(cuSparseMatrix_) {};

        ~sparseSingleGPUdata()
        {
            CHECK_CUSPARSE(cusparseDestroySpMat(this->cuSparseMatrix));
            if (this->ir_d) CHECK_CUDA(cudaFree(this->ir_d));
            if (this->pr_d) CHECK_CUDA(cudaFree(this->pr_d));
            if (this->jc_d) CHECK_CUDA(cudaFree(this->jc_d));
            CHECK_CUSPARSE(cusparseDestroy(this->cuSparseHandle));
            #ifdef NDEBUG
                mexPrintf("Device Memory freed and cusparse objects destroyed!\n");            
            #endif
        }
    };

    /// @brief construct an empty sparse matrix
    //sparseSingleGPU();

    /// @brief copy constructor for given Eigen matrix (mostly internal copies)
    /// @param cudaSpMatrix_ 
    sparseSingleGPU(std::shared_ptr<sparseSingleGPUdata> cudaSpMatrix_);

    /// @brief construct single sparse matrix from Matlab double sparse matrix
    /// @param sparseDouble 
    sparseSingleGPU(const mxArray *sparseDouble);

    /// @brief construct single sparse matrix from our custom single sparse matrix
    /// @param spSingle
    //sparseSingleGPU(const sparseSingle *spSingle);

    ~sparseSingleGPU();

    /// @brief Matrix/Vector product
    /// @param vals 
    /// @param n 
    /// @return a non-sparse single vector
    mxArray* timesVec(const mxSingle* vals,mwSize n, bool transposeInPlace = false) const;

    /// @brief Vector/Matrix product
    /// @param vals 
    /// @param n 
    /// @return a non-sparse single vector
    mxArray* vecTimes(const mxSingle* vals,mwSize n, bool transposeInPlace = false) const;

    mwSize getNnz() const;
    mwSize getCols() const;
    mwSize getRows() const;

    /// @brief Transposes the matrix by setting a transpose flag
    /// @return Pointer to transposed single sparse matrix. Will only hold a shared copy under the hood
    sparseSingleGPU* transpose() const;

    //// Indexing ////

    /// @brief Index with a row and column vector of indices
    /// @param rowIndex 
    /// @param colIndex 
    /// @return Pointer to sparse Single submatrix
    /// @todo This can be very slow. Better alternatives for slicing?
    //sparseSingle* rowColIndexing(const mxArray * const rowIndex, const mxArray * const colIndex) const;               
    

    /// @brief Return all values (when called from Maltab as A(:)) by just restructuring the matrix
    /// @return Pointer to sparseSingle (n*M)x1 matrix
    /// @todo Maybe we could also make a version that shares the data memory when read-only
    //sparseSingle* allValues() const;

    /// @brief display like Matlab's disp() function
    //void disp() const;

        
private:
    
    
    std::shared_ptr<sparseSingleGPUdata> cudaSpMatrix;  
    
    bool transposed = false; // Treats the matrix as transposed (we do not explicitly transpose it for performance)

    /// @brief Copy helper function
    /// @param devicePtrPtr pointer to pointer of device memory, such that the value is pointing to the correct location after call to the function
    /// @param hostPtr pointer to host memory
    /// @param n  nunber of elements to be copied
    /// @param stream cuda stream id, defaults to 0
    /// @return Pointer to new host array. nullptr when no cast has been done
    template<typename T, typename S, typename sz>
    T* cudaMallocAndMemcpyToDeviceWithCast(T** devicePtrPtr, S* hostPtr, const sz n, cudaStream_t stream = 0) const
    {
        CHECK_CUDA(cudaMallocAsync((void**)devicePtrPtr, n * sizeof(T),stream));

        bool casted = false;

        T* hostCast;
        if (std::is_same<T,S>::value)
            hostCast = (T*) hostPtr;
        else
        {
            hostCast = new T[n]; //Could use page locked memory (cudaMallocHost) here?
            std::transform(std::execution::par_unseq,hostPtr, hostPtr+n, hostCast, [](S d) -> T { return static_cast<T>(d);});            
            casted = true;
        }        
        
        CHECK_CUDA(cudaMemcpyAsync(*devicePtrPtr, hostCast, n * sizeof(T), cudaMemcpyHostToDevice,stream)); 
        
        if (casted)
            return hostCast;
        else
            return nullptr;
    }

        

};