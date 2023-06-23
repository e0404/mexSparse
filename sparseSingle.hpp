#ifndef SPARSE_SINGLE_HPP
#define SPARSE_SINGLE_HPP

#include <type_traits>
#include "mex.h"
#include <memory>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "class_handle.hpp"

template<mxClassID> struct mxClassToDataType_t;
template<> struct mxClassToDataType_t<mxClassID::mxDOUBLE_CLASS> { using type = mxDouble; };
template<> struct mxClassToDataType_t<mxClassID::mxSINGLE_CLASS> { using type = mxSingle; };

template<mxClassID T>
using mxClassToDataType = typename mxClassToDataType_t<T>::type;


class sparseSingle
{
public:    
    //// Typedefs ////
    typedef int64_t index_t;
    typedef Eigen::SparseMatrix<float,Eigen::ColMajor,index_t> spMat_t;
    typedef Eigen::SparseMatrix<float,Eigen::RowMajor,index_t> spMatTransposed_t;

    template<typename mxType>
    using mxAsMatrix_t = Eigen::Matrix<mxType,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;
    template<typename mxType>
    using mxAsArray_t = Eigen::Array<mxType,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;

    using mxSingleAsMatrix_t = mxAsMatrix_t<mxSingle>;
    using mxSingleAsArray_t = mxAsArray_t<mxSingle>;

    using mxDoubleAsMatrix_t = mxAsMatrix_t<mxDouble>;
    using mxDoubleAsArray_t = mxAsArray_t<mxDouble>;

    enum CscParallelism 
    {
        DEFAULT, // Leave this! The others are just for internal testing.
        WITHIN_COLUMN, 
        ACROSS_COLUMN
    };

    enum ElementWiseOperation
    {
        ELEMENTWISE_PLUS,
        ELEMENTWISE_TIMES,
        ELEMENTWISE_DIVIDE_L,
        ELEMENTWISE_DIVIDE_R,
        ELEMENTWISE_MINUS_R,
        ELEMENTWISE_MINUS_L
    };

    enum ElementWiseComparison
    {
        ELEMENTWISE_EQUAL,
        ELEMENTWISE_NOT_EQUAL,
        ELEMENTWISE_GREATER_EQUAL,
        ELEMENTWISE_GREATER,
        ELEMENTWISE_LOWER_EQUAL,
        ELEMENTWISE_LOWER
    };

    //// Functions ////

    /// @brief construct an empty sparse matrix
    sparseSingle();

    /// @brief copy constructor performing deep copy
    /// @param eigSpMatrix_ 
    sparseSingle(const sparseSingle& copy);

    

    /// @brief construct single sparse matrix from a matrix (matlab double sparse matrix, single matrix, or double matrix)
    /// @param sparseDouble 
    sparseSingle(const mxArray *inputMatrix);

    /// @brief construct empty sparse matrix with specific size
    /// @param m number of rows 
    /// @param n number of cols
    sparseSingle(const mxArray *m, const mxArray *n);

    /// @brief construct sparse matrix from triplets with size max(i)xmax(j)
    /// @param i row indices
    /// @param j col indices
    /// @param v values
    sparseSingle(const mxArray *i, const mxArray *j, const mxArray* v);

    /// @brief construct sparse matrix from triplets with given size
    /// @param i row indices 
    /// @param j col indices
    /// @param v values
    /// @param m number of rows 
    /// @param n number of cols
    /// @param nz space to reserve for nzs -- does this parameter make sense for our class? Maybe in horzcat/vertcat operations
    sparseSingle(const mxArray *i, const mxArray *j, const mxArray* v, const mxArray* m, const mxArray* n, const mxArray* nz = nullptr);

    ~sparseSingle();

    mwSize getNnz() const;
    mwSize getCols() const;
    mwSize getRows() const;
    mxArray* size() const;
    mxArray* nnz() const;
    bool isScalar() const;
    bool isSquare() const;

    /// @brief Transposes the matrix by setting a transpose flag
    /// @return Pointer to transposed single sparse matrix. Will only hold a shared copy under the hood
    mxArray* transpose() const;

    /// @brief Transposes the matrix by setting a transpose flag
    /// @return Pointer to transposed single sparse matrix. Will only hold a shared copy under the hood
    ///sparseSingle* setCscParallelism(const mxArray* cscParallelism_) const;
    
    /// @brief Transposes the matrix by setting a transpose flag
    /// @return Pointer to transposed single sparse matrix. Will only hold a shared copy under the hood
    ///mxArray* getCscParallelism() const;
    

    sparseSingle* horzcat() const;

    /// @brief Get dense matrix as mxArray
    /// @return Dense Matrix as mxArray
    mxArray* full() const;

    //// binary elementwise operations ////

    /// @brief Add a single matrix/scalar to sparse matrix
    /// @param summand single matrix as mxArray, can be also sparseSingle or scalar
    /// @return single matrix
    mxArray* plus(const mxArray* summand) const;

    /// @brief Add a single matrix/scalar to sparse matrix
    /// @param factor single matrix as mxArray, can be also sparseSingle or scalar
    /// @return single matrix
    mxArray* times(const mxArray* factor) const;

    /// @brief Returns the negated matrix
    /// @return single sparse matrix
    mxArray* uminus() const;

    /// @brief Subtract a sparse matrix from a single matrix/scalar
    /// @param minuend matrix as mxArray, can be also sparseSingle or scalar, sucht that result = minuend - this
    /// @return single matrix
    mxArray* minusAsSubtrahend(const mxArray* minuend) const;

    /// @brief Subtract a single matrix/scalar from a sparse matrix
    /// @param subtrahend matrix as mxArray, can be also sparseSingle or scalar, sucht that result = this - subtrahend
    /// @return single matrix
    mxArray* minusAsMinuend(const mxArray* subtrahend) const;

    /// @brief Divide a sparse matrix by given divisor
    /// @param divisor matrix as mxArray, can be also sparseSingle or scalar, sucht that result = this./divisor
    /// @return single matrix
    mxArray* rdivide(const mxArray* divisor) const;

    /// @brief Divide a dividend by a sparse matrix
    /// @param dividend matrix as mxArray, can be also sparseSingle or scalar, sucht that result = dividend./this
    /// @return single matrix
    mxArray* ldivide(const mxArray* dividend) const;

    /// LINEAR ALGEBRA ///

    /// @brief Matrix multiplication from the right to this matrix
    /// @param rightfactor pointer to mxArray containt the right factor to (matrix) multiply 
    /// @return a dense matrix or a sparse single handle as mxArray
    mxArray* mtimesr(const mxArray* rightfactor) const;

    /// @brief Matrix multiplication from the left to this matrix
    /// @param rightfactor pointer to mxArray containt the left factor to (matrix) multiply 
    /// @return a dense matrix or a sparse single handle as mxArray
    mxArray* mtimesl(const mxArray* leftfactor) const;

    /// @brief Solve the system Ax = b for x
    /// @param b constant equality matrix as mxArray, can be also sparseSingle or scalar, sucht that result = this\b
    /// @return single matrix (solution x)
    mxArray* mldivide(const mxArray* b) const;

    /// @brief Matrix/Vector product
    /// @param vals 
    /// @param n 
    /// @return a non-sparse single vector
    mxArray* timesVec(const mxArray* vals) const;

    /// @brief Vector/Matrix product
    /// @param vals 
    /// @param n 
    /// @return a non-sparse single vector
    mxArray* vecTimes(const mxArray* vals) const;

    /// @brief Multiplication with Scalar
    /// @param scalar
    /// @return scaled sparse single matrix
    mxArray* timesScalar(const mxArray* val) const;

    //// Indexing ////

    /// @brief Index with a row and column vector of indices
    /// @param rowIndex 
    /// @param colIndex 
    /// @return Pointer to sparse Single submatrix
    /// @todo This can be very slow. Better alternatives for slicing?
    mxArray* rowColIndexing(const mxArray * const rowIndex, const mxArray * const colIndex) const;  

    /// @brief Row / Column Index assignment
    /// @param rowIndex 
    /// @param colIndex 
    mxArray* rowColAssignment(const mxArray * const rowIndex, const mxArray * const colIndex, const mxArray* assignedValues);                 
    
    /// @brief Index with a linear index list
    /// @param indexList
    /// @return Pointer to sparse Single submatrix (will have a vector dimension)
    /// @todo Not fully implemented yet
    mxArray* linearIndexing(const mxArray* indexList) const;    
    
    /// @brief return all nonzero values
    /// @return Pointer to column array of linear indices
    mxArray* find() const;

    /// @brief display like Matlab's disp() function
    void disp() const;   

        
private:
    /// MEMBER VARIABLES ////
    std::shared_ptr<spMat_t> eigSpMatrix; //The matrix is stored as shared pointer to allow management of copies, such that we sensibly use a matlab data class instead of a handle
    bool transposed = false; // Treats the matrix as transposed (we do not explicitly transpose it for performance)
    
    const CscParallelism cscParallelize = DEFAULT; // Leave this! The others are just for internal testing.
    
    //// PRIVATE MEMBER FUNCTIONS ////
    /// @brief non-copy constructor for given Eigen matrix (mostly internal copies)
    /// @param eigSpMatrix_ 
    sparseSingle(std::shared_ptr<spMat_t> eigSpMatrix_);

    /// @brief copy constructor sharing matrix storage
    /// @param eigSpMatrix_ 
    sparseSingle(sparseSingle& shared_copy);

    /// @brief Return all values (when called from Maltab as A(:)) by just restructuring the matrix
    /// @return Pointer to sparseSingle (n*M)x1 matrix
    /// @todo Maybe we could also make a version that shares the data memory when read-only
    sparseSingle* allValues() const;

    /*
    template<typename T>    
    inline mxArray* mtimesl_typedMultiplier(mxArray* leftFactor, T tmp = T()) const
    {
        using Tvalue = mxClassToDataType<T>;
        mwSize m = mxGetM(leftFactor);
        mxArray* resultMatrix = mxCreateNumericMatrix(m,this->getCols(),mxSINGLE_CLASS,mxREAL);
        mxSingle* result_data = mxGetSingles(resultMatrix);
        Eigen::Map<mxSingleAsMatrix_t> resultMap(result_data,m,this->getCols());

        //Create a Map to the Eigen vector
        Tvalue* vals;
        if (std::is_same<mxSingle,Tvalue>)
            vals = mxGetSingles(leftFactor);
        else if (std::is_same<mxSingle,Tvalue>)
            vals = mxGetDoubles(leftFactor);
        else
            throw(MexException("sparseSingle:wrongDataType"),"Data type not supported!");
        
        mxAsMatrix_t<Tvalue> factorMatrixMap(vals,m,n);

        if (this->transposed)
            resultMap = factorMatrixMap.cast<float>() * this->eigSpMatrix->transpose();
        else
            resultMap = factorMatrixMap.cast<float>() * (*this->eigSpMatrix); 

        return resultMatrix;
    }
    */


    /// @brief construct sparse matrix from triplets with given size
    /// @param i row indices 
    /// @param j col indices
    /// @param v values
    /// @param m number of rows 
    /// @param n number of cols
    /// @param nz space to reserve for nzs -- does this parameter make sense for our class? Maybe in horzcat/vertcat operations
    void constructFromMatlabTriplets(const mxArray *i, const mxArray *j, const mxArray* v, const mxArray* m, const mxArray* n, const mxArray* nz = nullptr);

    void reportSolverInfo(Eigen::ComputationInfo& info) const;
    
    index_t toLinearIndex(const index_t row, const index_t col) const;

    index_t linearIndexToColIndex(const index_t linIx) const;

    index_t linearIndexToRowIndex(const index_t linIx) const;

    mxArray* elementWiseBinaryOperation(const mxArray* operand, const ElementWiseOperation& op) const;

    mxArray* elementWiseComparison(const mxArray* operand, const ElementWiseComparison& op) const;

    template<typename T>
    bool isConsecutiveArray(const T * const array, const index_t n) const
    {
        for(index_t i = 1; i < n; ++i)
        {
            if(array[i] != array[i-1]+1)
            {
                return false;
            }
        }
        return true;
    }

    

    class Matlab2EigenIndexListConverter 
    {       
    public:

        sparseSingle::index_t size() const { return index_t(this->nIndices); }
        sparseSingle::index_t operator[] (sparseSingle::index_t i) const { return static_cast<index_t>(this->indexData[i]) - 1; }        

        const double *const data() const { return this->indexData; }

        Matlab2EigenIndexListConverter(const mxArray * const ixList)
        {
            if (mxIsDouble(ixList)) //Further conditions?
            {
                this->indexData = mxGetPr(ixList);
                this->nIndices = mxGetNumberOfElements(ixList);
            }
            else
                throw(MexException("sparseSingle::invalidIndex","Invalid Index List"));
        }

    private:
        const double * indexData;
        mwSize  nIndices;
    };

    template<typename T>
    class UntypedMxDataAccessor
    {
    public:
        size_t size() const { return this->numElements; }

        T operator[](size_t i) const 
        {
            switch(this->classID)
            {
                case mxDOUBLE_CLASS:
                    return static_cast<T>( ((mxDouble*) this->data)[i] );
                case mxSINGLE_CLASS:
                    return static_cast<T>( ((mxSingle*) this->data)[i] );
                case mxINT8_CLASS:
                    return static_cast<T>( ((mxInt8*) this->data)[i] );                
                case mxINT16_CLASS:
                    return static_cast<T>( ((mxInt16*) this->data)[i] );
                case mxINT32_CLASS:
                    return static_cast<T>( ((mxInt32*) this->data)[i] );
                case mxINT64_CLASS:
                    return static_cast<T>( ((mxInt64*) this->data)[i] );                
                case mxUINT8_CLASS:
                    return static_cast<T>( ((mxUint8*) this->data)[i] );                
                case mxUINT16_CLASS:
                    return static_cast<T>( ((mxUint16*) this->data)[i] );
                case mxUINT32_CLASS:
                    return static_cast<T>( ((mxUint32*) this->data)[i] );
                case mxUINT64_CLASS:
                    return static_cast<T>( ((mxUint64*) this->data)[i] ); 
                default:
                    throw(MexException("sparseSingle:invalidDataType","Invalid Data Type in mxArray!"));
            };            
        }

        UntypedMxDataAccessor(const mxArray* const dataArray) 
        {
            if(dataArray == nullptr)
                throw(MexException("sparseSingle:invalidDataPointer","Data pointer to mxArray was empty!"));
            
            if(!mxIsNumeric(dataArray) || mxIsSparse(dataArray))
                throw(MexException("sparseSingle:invalidDataPointer","Invalid Data is mxArray!"));

            this->classID = mxGetClassID(dataArray);            
            this->data = mxGetData(dataArray);
            this->numElements = mxGetNumberOfElements(dataArray);
        }

        

    private:
        mxClassID classID;
        const void* data;
        size_t numElements;

    };
};

#endif //SPARSE_SINGLE_HPP