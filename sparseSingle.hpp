#pragma once

#include "mex.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <exception>
#include <string>

class MexException : public std::exception
{
    public:       
        explicit MexException(const std::string& matlabErrID, const std::string& matlabErrMessage)
            : errID(matlabErrID), errMessage(matlabErrMessage) {}

        virtual ~MexException() noexcept {}

        virtual const char* what() const override
        {                        
            return this->errMessage.c_str();
        }

        virtual const char* id() const noexcept
        {
            return this->errID.c_str();
        }

    protected:
        std::string errID;
        std::string errMessage;
}; 

class sparseSingle
{
public:    
    //// Typedefs ////
    typedef int64_t index_t;
    typedef Eigen::SparseMatrix<float,Eigen::ColMajor,index_t> spMat_t;
    typedef Eigen::SparseMatrix<float,Eigen::RowMajor,index_t> spMatTransposed_t;

    typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> mxSingleAsMatrix_t;
    typedef Eigen::Array<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> mxSingleAsArray_t;
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> mxDoubleAsMatrix_t;
    typedef Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> mxDoubleAsArray_t;

    enum CscParallelism 
    {
        DEFAULT, // Leave this! The others are just for internal testing.
        WITHIN_COLUMN, 
        ACROSS_COLUMN
    };

    //// Functions ////

    /// @brief construct an empty sparse matrix
    sparseSingle();

    /// @brief copy constructor for given Eigen matrix (mostly internal copies)
    /// @param eigSpMatrix_ 
    sparseSingle(std::shared_ptr<spMat_t> eigSpMatrix_);

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
    sparseSingle* timesScalar(const mxArray* val) const;

    mwSize getNnz() const;
    mwSize getCols() const;
    mwSize getRows() const;
    mxArray* size() const;
    mxArray* nnz() const;

    /// @brief Transposes the matrix by setting a transpose flag
    /// @return Pointer to transposed single sparse matrix. Will only hold a shared copy under the hood
    sparseSingle* transpose() const;

    /// @brief Transposes the matrix by setting a transpose flag
    /// @return Pointer to transposed single sparse matrix. Will only hold a shared copy under the hood
    ///sparseSingle* setCscParallelism(const mxArray* cscParallelism_) const;
    
    /// @brief Transposes the matrix by setting a transpose flag
    /// @return Pointer to transposed single sparse matrix. Will only hold a shared copy under the hood
    ///mxArray* getCscParallelism() const;
    

    /// @brief Get dense matrix as mxArray
    /// @return Dense Matrix as mxArray
    mxArray* full() const;

    /// @brief Add a dense single matrix to sparse matrix
    /// @param denseMx single matrix as mxArray
    /// @return Dense single matrix
    mxArray* addDense(const mxArray* denseMx) const;

    //// Indexing ////

    /// @brief Index with a row and column vector of indices
    /// @param rowIndex 
    /// @param colIndex 
    /// @return Pointer to sparse Single submatrix
    /// @todo This can be very slow. Better alternatives for slicing?
    sparseSingle* rowColIndexing(const mxArray * const rowIndex, const mxArray * const colIndex) const;               
    
    /// @brief Index with a linear index list
    /// @param indexList
    /// @return Pointer to sparse Single submatrix (will have a vector dimension)
    /// @todo Not fully implemented yet
    sparseSingle* linearIndexing(const mxArray* indexList) const;    

    /// @brief Return all values (when called from Maltab as A(:)) by just restructuring the matrix
    /// @return Pointer to sparseSingle (n*M)x1 matrix
    /// @todo Maybe we could also make a version that shares the data memory when read-only
    sparseSingle* allValues() const;

    /// @brief display like Matlab's disp() function
    void disp() const;   

        
private:
    /// MEMBER VARIABLES ////
    std::shared_ptr<spMat_t> eigSpMatrix; //The matrix is stored as shared pointer to allow management of copies, such that we sensibly use a matlab data class instead of a handle
    bool transposed = false; // Treats the matrix as transposed (we do not explicitly transpose it for performance)
    
    const CscParallelism cscParallelize = DEFAULT; // Leave this! The others are just for internal testing.
    

    //// PRIVATE MEMBER FUNCTIONS ////

    /// @brief construct sparse matrix from triplets with given size
    /// @param i row indices 
    /// @param j col indices
    /// @param v values
    /// @param m number of rows 
    /// @param n number of cols
    /// @param nz space to reserve for nzs -- does this parameter make sense for our class? Maybe in horzcat/vertcat operations
    void constructFromMatlabTriplets(const mxArray *i, const mxArray *j, const mxArray* v, const mxArray* m, const mxArray* n, const mxArray* nz = nullptr);
    
    index_t toLinearIndex(const index_t row, const index_t col) const;

    index_t linearIndexToColIndex(const index_t linIx) const;

    index_t linearIndexToRowIndex(const index_t linIx) const;


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