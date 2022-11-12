#include "sparseSingle.hpp"

#include <algorithm>
#include <execution>
#include <chrono>
#include <array>

//// Construct & Delete ////

sparseSingle::sparseSingle() 
{ 
    this->eigSpMatrix = std::make_shared<spMat_t>();
}

sparseSingle::sparseSingle(std::shared_ptr<spMat_t> eigSpMatrix_) 
{
        this->eigSpMatrix = eigSpMatrix_;        
    }

sparseSingle::sparseSingle(const mxArray *inputMatrix) 
{
        if (!inputMatrix)
        {
            throw(MexException("sparseSingle:invalidInputType","Matrix to construct from invalid!"));         
        }        
        if (mxIsSparse(inputMatrix) && mxIsDouble(inputMatrix)) //I think there's also sparse logicals
        {
            mwIndex *ir, *jc; // ir: row indec, jc: encode row index and values in pr per coloumn
            double *pr; //value pointer
            
            // Get the starting pointer of all three data arrays.
            pr = mxGetPr(inputMatrix);     // row index array
            ir = mxGetIr(inputMatrix);     // row index array
            jc = mxGetJc(inputMatrix);     // column encrypt array
            mwSize nCols = mxGetN(inputMatrix);       // number of columns
            mwSize nRows = mxGetM(inputMatrix);       // number of rows

            // nnz = mxGetNzmax(prhs[0]); // number of possible non zero elements
            mwSize nnz = jc[nCols]; // number of non zero elements currently stored inside the sparse matrix
            
            //Create the Eigen Sparse Matrix        
            try {        
                //this->eigSpMatrix = std::shared_ptr<spMat_t>(new spMat_t(nRows,nCols));
                this->eigSpMatrix = std::make_shared<spMat_t>(nRows,nCols);
                this->eigSpMatrix->makeCompressed();
                this->eigSpMatrix->reserve(nnz);
                std::transform(std::execution::par_unseq, pr, pr+nnz, this->eigSpMatrix->valuePtr(), [](double d) -> float { return static_cast<float>(d);});    
                std::transform(std::execution::par_unseq, ir, ir+nnz, this->eigSpMatrix->innerIndexPtr(), [](mwIndex i) -> index_t { return static_cast<index_t>(i);});
                std::transform(std::execution::par_unseq, jc, jc+(nCols+1), this->eigSpMatrix->outerIndexPtr(), [](mwIndex i) -> index_t { return static_cast<index_t>(i);});
            }
            catch (const std::exception& e) {
                std::string msg = std::string("Eigen Map could not be constructed from sparse matrix! Caught exception ") + e.what();      
                throw(MexException("sparseSingle:errorOnConstruct",msg));
            }
            catch (...)
            {
                throw(MexException("sparseSingle:errorOnConstruct","Eigen Map could not be constructed from sparse matrix!"));
            }
                   
        }
        else if (mxIsSingle(inputMatrix)) // full matrix
        {
            mwSize nCols = mxGetN(inputMatrix);       // number of columns
            mwSize nRows = mxGetM(inputMatrix);       // number of rows
            float *singleData = mxGetSingles(inputMatrix);

            Eigen::Map<mxSingleAsMatrix_t> singleDataMap(singleData,nRows,nCols);            
            this->eigSpMatrix = std::make_shared<spMat_t>(singleDataMap.sparseView());
        }
        else if (mxIsDouble(inputMatrix))
        {
            mwSize nCols = mxGetN(inputMatrix);       // number of columns
            mwSize nRows = mxGetM(inputMatrix);       // number of rows
            double *doubleData = mxGetDoubles(inputMatrix);

            Eigen::Map<mxDoubleAsMatrix_t> singleDataMap(doubleData,nRows,nCols);            
            this->eigSpMatrix = std::make_shared<spMat_t>(singleDataMap.cast<float>().sparseView());
        }
        else
        {
            throw(MexException("sparseSingle:invalidInputType","Invalid Input Argument!"));      
        }
        this->eigSpMatrix->makeCompressed(); // Not sure if necessary
    }

sparseSingle::sparseSingle(const mxArray *m_, const mxArray *n_) 
{
    //Argument checks
    if (!mxIsScalar(m_) || !mxIsScalar(n_))
        throw(MexException("sparseSingle:invalidInputType","Row and Column Number must both be scalars!"));
    
    if ((!mxIsNumeric(m_) && !mxIsChar(m_) ) || !mxIsNumeric(n_) && !mxIsChar(n_))
        throw(MexException("sparseSingle:invalidInputType","Row and/or Column Number input is invalid!"));
    
    //Note that this implicitly casts to double and thus also allows other data types from matlab
    index_t m = (index_t) mxGetScalar(m_); 
    index_t n = (index_t) mxGetScalar(n_);

    this->eigSpMatrix = std::make_shared<spMat_t>(m,n);
}

sparseSingle::sparseSingle(const mxArray* i_, const mxArray* j_, const mxArray* v_)
{
    //We only obtain the size here before calling the construction with given sizes
    index_t maxI = 0;
    index_t maxJ = 0;
    UntypedMxDataAccessor<index_t> i(i_);
    UntypedMxDataAccessor<index_t> j(j_);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            
            for (size_t n=0; n < i.size(); n++)
                maxI = std::max<index_t>(maxI,i[n]);
        }
        
        #pragma omp section
        {
            
            for (size_t n=0; n < j.size(); n++)
                maxJ = std::max<index_t>(maxJ,j[n]);
        }
    }

    mxArray* m = mxCreateDoubleScalar((double) maxI);
    mxArray* n = mxCreateDoubleScalar((double) maxJ);

    this->constructFromMatlabTriplets(i_,j_,v_,m,n);

    mxDestroyArray(m);
    mxDestroyArray(n);
}

sparseSingle::sparseSingle(const mxArray* i, const mxArray* j, const mxArray* v, const mxArray* m, const mxArray* n, const mxArray* nz)
{
    this->constructFromMatlabTriplets(i,j,v,m,n,nz);
}

void sparseSingle::constructFromMatlabTriplets(const mxArray* i_, const mxArray* j_, const mxArray* v_, const mxArray* m_, const mxArray* n_, const mxArray* nz_)
{
    //We fill triplets manually because Eigen would expect them as a Triplet construct, but we have independent mxArrays and would need to copy everything together

    UntypedMxDataAccessor<index_t> i(i_);
    UntypedMxDataAccessor<index_t> j(j_);

    //For now we mimic the SparseDouble behavior of only allowing values of similar type (here singles). We could cast, if we want to, as well  
    if (v_ == nullptr || !mxIsSingle(v_))
        throw(MexException("sparseSingle:invalidInputType","Values must be of data type single"));

    mwSize numValues = mxGetNumberOfElements(v_); //We can even have matrices as input, so we only care for the number of elements
    float* v = mxGetSingles(v_);    

    if ((i.size() != j.size()) || j.size() != numValues)
        throw(MexException("sparseSingle:invalidInputType","Different number of elements in input triplet vectors!"));
    
    std::vector<index_t> sortPattern(numValues);
    #pragma omp parallel for schedule(static)
    for (index_t r = 0; r < numValues; r++)
        sortPattern[r] = r;

    if (!mxIsScalar(m_) || !mxIsScalar(n_) || !mxIsNumeric(m_) || !mxIsNumeric(n_))
        throw(MexException("sparseSingle:invalidInputType","Row and Column numbers must be numeric scalars!"));

    index_t m = mxGetScalar(m_);
    index_t n = mxGetScalar(n_);

    if (m < 0 || n < 0)
        throw(MexException("sparseSingle:invalidInputType","Row and Column numbers must be greater or equal to zero!"));

    index_t nnz_reserve = numValues;
    if (nz_ != nullptr)
    {
        if(!mxIsScalar(nz_) || !mxIsNumeric(nz_))
            throw(MexException("sparseSingle:invalidInputType","Invalid number of nonzeros to reserve"));
        
        nnz_reserve = (index_t) mxGetScalar(nz_);

        //Should we throw an error here or just silently adapt?
        if (nnz_reserve < numValues)
            nnz_reserve = numValues;
    }        

    if (m < 0 || n < 0)
        throw(MexException("sparseSingle:invalidInputType","Row and Column numbers must be greater or equal to zero!"));

    //Now we obtain the sort pattern of the triplets
    //The data accessor is not yet in index base 0!!
    //We could use a linear index comparison or we define a double comparison
    //We sort by linear index
    std::stable_sort(std::execution::par,sortPattern.begin(),sortPattern.end(),
        [&i,&j,&m](index_t i1, index_t i2) {
            return ((j[i1]-1)*m + i[i1] - 1) < ((j[i2]-1)*m + i[i2] - 1);
        });
    
    this->eigSpMatrix = std::make_shared<spMat_t>(m,n);
    this->eigSpMatrix->reserve(nnz_reserve);
    this->eigSpMatrix->outerIndexPtr()[0] = 0;

    //Can this be parallelized if the indices are sorted as in our case?
    //#pragma omp parallel for schedule(static)
    for (index_t r = 0; r < numValues; r++)
    {
        //This is the index to be written
        index_t getIx = sortPattern[r];

        //Convert to base 0
        index_t row = i[getIx] - 1;
        index_t col = j[getIx] - 1;
        float value = v[getIx];

        //mexPrintf("Inserting triplet %d at (%d,%d) with value %f;.\n",r,row,col,value);

        this->eigSpMatrix->innerIndexPtr()[r] = row;
        this->eigSpMatrix->valuePtr()[r] = value;
        this->eigSpMatrix->outerIndexPtr()[col+1]++;
    }
    std::partial_sum(this->eigSpMatrix->outerIndexPtr(),this->eigSpMatrix->outerIndexPtr()+n+1,this->eigSpMatrix->outerIndexPtr());

    this->eigSpMatrix->makeCompressed();
}

 sparseSingle::~sparseSingle()
 {
    #ifndef NDEBUG 
        mexPrintf("Calling destructor - %d single sparse matrix instances still exist!\n",this->eigSpMatrix.use_count() - 1);
    #endif
}


//// Getters & Setters ////

mwSize sparseSingle::getNnz() const {
    return this->eigSpMatrix->nonZeros();
}

mwSize sparseSingle::getCols() const {
    if (this->transposed)
        return this->eigSpMatrix->rows();
    else
        return this->eigSpMatrix->cols();
}

mwSize sparseSingle::getRows() const {
    if (this->transposed)
        return this->eigSpMatrix->cols();
    else
        return this->eigSpMatrix->rows();
}

mxArray* sparseSingle::size() const {
    mxArray* szArray = mxCreateDoubleMatrix(1,2,mxREAL);
    double* pr = mxGetDoubles(szArray);
    pr[0] = static_cast<double>(this->getRows());
    pr[1] = static_cast<double>(this->getCols());
    return szArray;
}

mxArray* sparseSingle::nnz() const {
    return mxCreateDoubleScalar((double) this->getNnz());
}

//// Indexing ////
 
sparseSingle* sparseSingle::rowColIndexing(const mxArray * const rowIndex, const mxArray * const colIndex) const
{
    //TODO: Transpose Implementation
    
    sparseSingle* indexedSubMatrix = nullptr;

    //Check if we are indexing a block
    bool consecutiveRows = false;
    bool consecutiveCols = false;

    sparseSingle::Matlab2EigenIndexListConverter rowIndices4Eigen(rowIndex);
    sparseSingle::Matlab2EigenIndexListConverter colIndices4Eigen(colIndex);

    const double * const rowIndexData = rowIndices4Eigen.data();
    const double * const colIndexData = colIndices4Eigen.data();

    const index_t nRowIndices = rowIndices4Eigen.size();
    const index_t nColIndices = colIndices4Eigen.size(); 

    #pragma omp parallel sections
    {   
        #pragma omp section
        consecutiveRows = this->isConsecutiveArray(rowIndexData,nRowIndices);
        #pragma omp section
        consecutiveCols = this->isConsecutiveArray(colIndexData,nColIndices);
    }
    
    bool blockIndexing = consecutiveRows & consecutiveCols;

    //Debug output
    #ifndef NDEBUG
        mexPrintf("Block indexing detected? %s\n",blockIndexing ? "true" : "false");
    #endif

    if (blockIndexing)
    {
        if(this->transposed)
        {
            //mexErrMsgTxt("Transpose not implemented!");
            throw(MexException("sparseSingle:implementationMissing","Transpose not implemented!"));
        }
        else{
            
            index_t startRow = rowIndices4Eigen[0];
            index_t rows = nRowIndices;

            index_t startCol = colIndices4Eigen[0];
            index_t cols = nColIndices;

        
            auto block = this->eigSpMatrix->block(startRow,startCol,rows,cols);
            std::shared_ptr<spMat_t> blockSpMat = std::make_shared<spMat_t>(block);
            
            indexedSubMatrix = new sparseSingle(blockSpMat);
        }
        
    }
    else
    {
        //Eigen Supports slicing for Dense Matrices Only, so we need to manually slice the matrix
        //Indexing by Matrix Multiplication as found here: https://people.eecs.berkeley.edu/~aydin/spgemm_sisc12.pdf
        // Matlab equivalent
        // [m,n] = size(A);
        // R = sparse(1:len(I),I,1,len(I),m);
        // Q = sparse(J,1:len(J),1,n,len(J));
        // B=R*A*Q;

        if(this->transposed)
        {
            throw(MexException("sparseSingle:implementationMissing","Transpose not implemented!"));
        }
        else
        {            
            typedef Eigen::Triplet<float_t,index_t> T;
            spMat_t R(nRowIndices,this->getRows());
            spMat_t Q(this->getCols(),nColIndices);
            //Build the R matrix

            std::vector<T> tripletListR(nRowIndices);
            //tripletListR.reserve(nRowIndices);
            #pragma omp parallel for schedule(static)
            for (index_t i = 0; i < nRowIndices; i++)
                tripletListR[i] = T(i,rowIndices4Eigen[i],1);      
            R.setFromTriplets(tripletListR.begin(),tripletListR.end()) ;

            //Build the Q matrix
            std::vector<T> tripletListQ(nColIndices);
            //tripletListQ.reserve(nColIndices);
            #pragma omp parallel for schedule(static)
            for (index_t j = 0; j < nColIndices; j++)
                tripletListQ[j] = T(colIndices4Eigen[j],j,1);              
            Q.setFromTriplets(tripletListQ.begin(),tripletListQ.end());                       

            //Now perform the slicing product
            std::shared_ptr<spMat_t> subSpMat = std::make_shared<spMat_t>(nRowIndices,nColIndices);
            //subSpMat->makeCompressed();
            (*subSpMat) = R*(*this->eigSpMatrix)*Q;
            indexedSubMatrix = new sparseSingle(subSpMat); 
        }
    }

    return indexedSubMatrix; 
}

sparseSingle* sparseSingle::allValues() const 
{
    index_t numValues = this->getRows()*this->getCols();
    index_t nnz = this->getNnz();
    //mxArray* result = mxCreateNumericMatrix(numValues,1,mxSINGLE_CLASS,mxREAL);
    //mxSingle* result_data = mxGetSingles(result);

    std::shared_ptr<spMat_t> subSpMat = std::make_shared<spMat_t>(numValues,1);
    subSpMat->reserve(nnz); 

    //Sanity Check
    if (!(this->eigSpMatrix->isCompressed()))
        throw(MexException("sparseSingle:invalidMatrixState","The matrix is not compressed! This is unexpected behavior!"));

    std::copy(std::execution::par_unseq,this->eigSpMatrix->valuePtr(),this->eigSpMatrix->valuePtr() + nnz, subSpMat->valuePtr());
    subSpMat->outerIndexPtr()[0] = index_t(0);
    subSpMat->outerIndexPtr()[1] = nnz;


    if (this->transposed)  
    {
        Eigen::Map<spMatTransposed_t> crs_transposed(this->getRows(),this->getCols(),this->getNnz(),this->eigSpMatrix->outerIndexPtr(),this->eigSpMatrix->innerIndexPtr(),this->eigSpMatrix->valuePtr());
        index_t count = 0;
        //#pragma omp parallel for schedule(dynamic)
        for (index_t k = 0; k < crs_transposed.outerSize(); ++k)
            for (Eigen::Map<spMatTransposed_t>::InnerIterator it(crs_transposed,k); it; ++it)
            {
                index_t linearIndex = this->toLinearIndex(it.row(),it.col());
                subSpMat->innerIndexPtr()[count] = linearIndex;
                count++;
            }
    }
    else{
        #pragma omp parallel for schedule(dynamic)
        for (index_t k = 0; k < this->eigSpMatrix->outerSize(); ++k)
        {
            index_t nnzInCol = this->eigSpMatrix->outerIndexPtr()[k+1] - this->eigSpMatrix->outerIndexPtr()[k];
            index_t offset = this->eigSpMatrix->outerIndexPtr()[k];
            index_t count = 0;

            for (spMat_t::InnerIterator it(*this->eigSpMatrix,k); it; ++it)
            {
                index_t linearIndex = this->toLinearIndex(it.row(),it.col());
                subSpMat->innerIndexPtr()[offset + count] = linearIndex;
                count++;
            }
        }               
    }                      

    sparseSingle* indexedMatrix = new sparseSingle(subSpMat);

    return indexedMatrix;
}

void sparseSingle::disp() const 
{
    if (this->getNnz() == 0)
    {
        mexPrintf("   All zero sparse single: %dx%d\n",this->getRows(),this->getCols());
    }            

    if (this->transposed)  
    {
        Eigen::Map<spMatTransposed_t> crs_transposed(this->getRows(),this->getCols(),this->getNnz(),this->eigSpMatrix->outerIndexPtr(),this->eigSpMatrix->innerIndexPtr(),this->eigSpMatrix->valuePtr());

        for (index_t k = 0; k < crs_transposed.outerSize(); ++k)
            for (Eigen::Map<spMatTransposed_t>::InnerIterator it(crs_transposed,k); it; ++it)
            {
                mexPrintf("\t(%d,%d)\t\t%g\n",it.row()+1,it.col()+1,it.value());
            }
    }
    else
    {
        for (index_t k = 0; k < this->eigSpMatrix->outerSize(); ++k)
            for (spMat_t::InnerIterator it(*this->eigSpMatrix,k); it; ++it)
            {
                mexPrintf("\t(%d,%d)\t\t%g\n",it.row()+1,it.col()+1,it.value());
            }
    }
}

sparseSingle* sparseSingle::linearIndexing(const mxArray* indexList) const 
{
    //First check if it is indeed an index list or a colon operator
    mxClassID ixType = mxGetClassID(indexList);

    sparseSingle* result = nullptr;

    if (ixType == mxCHAR_CLASS) // We have a colon operator
        result = this->allValues();
    else if (ixType == mxDOUBLE_CLASS)
    {
        //mexErrMsgTxt("Only colon indexing supported at the moment!");
        
        
        //Normal double indexing list
        mwSize nDim = mxGetNumberOfDimensions(indexList);
        if (nDim > 2)
            throw(MexException("sparseSingle:invalidIndex","Indexing list has dimensionality bigger than 2!"));
    
        const mwSize* ixDim = mxGetDimensions(indexList);

        index_t numValues;
        bool isColumnVector;

        if (ixDim[0] == 1)
        {
            numValues = mxGetN(indexList);
            isColumnVector = false;
        }
        else if (ixDim[1] == 1)
        {
            numValues = mxGetM(indexList);
            isColumnVector = true;
        }
        else
            throw(MexException("sparseSingle:implementationMissing","Only vector index lists are implemented for now!"));

        index_t nnz = this->getNnz();

        std::shared_ptr<spMat_t> subSpMat;
        
        Matlab2EigenIndexListConverter indexList4Eigen(indexList);

        //check first if we have a scalar to avoid expensive copies
        if (numValues == 1)
        {
            index_t linearIndex = indexList4Eigen[0];
            index_t rowIx = this->linearIndexToRowIndex(linearIndex);
            index_t colIx = this->linearIndexToColIndex(linearIndex);
            float value;
            if (!this->transposed)
                value = this->eigSpMatrix->coeff(rowIx,colIx);
            else
                value = this->eigSpMatrix->coeff(colIx,rowIx);
            
            subSpMat = std::make_shared<spMat_t>(1,1);
            if (value > 0.0)
                subSpMat->coeffRef(0,0) = value;

            isColumnVector = true; //We do not want to set the transpose flag in this case at any times
        }
        else{
            
            std::vector<sparseSingle::index_t> tmpInnerIndex(nnz);
            std::array<sparseSingle::index_t,2> tmpOuterIndex;
            tmpOuterIndex[0] = 0;
            tmpOuterIndex[1] = nnz;
            if (this->transposed)  
            {
                Eigen::Map<spMatTransposed_t> crs_transposed(this->getRows(),this->getCols(),nnz,this->eigSpMatrix->outerIndexPtr(),this->eigSpMatrix->innerIndexPtr(),this->eigSpMatrix->valuePtr());
                index_t count = 0;
                //#pragma omp parallel for schedule(dynamic)
                for (index_t k = 0; k < crs_transposed.outerSize(); ++k)
                    for (Eigen::Map<spMatTransposed_t>::InnerIterator it(crs_transposed,k); it; ++it)
                    {
                        index_t linearIndex = this->toLinearIndex(it.row(),it.col());
                        tmpInnerIndex[count] = linearIndex;
                        count++;
                    }
            }
            else{
                //#pragma omp parallel for schedule(dynamic)
                for (index_t k = 0; k < this->eigSpMatrix->outerSize(); ++k)
                {
                    index_t nnzInCol = this->eigSpMatrix->outerIndexPtr()[k+1] - this->eigSpMatrix->outerIndexPtr()[k];
                    index_t offset = this->eigSpMatrix->outerIndexPtr()[k];
                    index_t count = 0;

                    for (spMat_t::InnerIterator it(*this->eigSpMatrix,k); it; ++it)
                    {
                        index_t linearIndex = this->toLinearIndex(it.row(),it.col());
                        tmpInnerIndex[offset + count] = linearIndex;
                        count++;
                    }
                }               
            }

            //typedef Eigen::SparseVector<float,Eigen::ColMajor,index_t> spColVec_t;
            Eigen::Map<spMat_t> spMatAsVector(this->getRows()*this->getCols(),1,nnz,tmpOuterIndex.data(),tmpInnerIndex.data(),this->eigSpMatrix->valuePtr());

            //Check if we have a range
            bool isRange = this->isConsecutiveArray(indexList4Eigen.data(),numValues);

            if (isRange)
            {   //mexPrintf("We have a block!\n");
                index_t start = indexList4Eigen[0];
        
                auto block = spMatAsVector.block(start,0,numValues,1);
                subSpMat = std::make_shared<spMat_t>(block);
            }
            else 
            {   
                typedef Eigen::Triplet<float,index_t> T;
                std::vector<T> triplets;

                std::vector<index_t> sortPattern(numValues);
                #pragma omp parallel for schedule(static)
                for (index_t i = 0; i < numValues; i++)
                    sortPattern[i] = i;

                std::stable_sort(std::execution::par,sortPattern.begin(),sortPattern.end(),[&indexList4Eigen](index_t i1, index_t i2) {return indexList4Eigen[i1] < indexList4Eigen[i2];});
                
                index_t searchIxSpVec = 0;
                index_t searchInnerIndex;

                //Now perform the search and exploit the sorting of the index string
                //We accumulate triplets since it is difficult to know beforehand how much storage we need. 
                //Alternatively, we could directly fill the column vector sparse storage and then perform a 
                //sort on the vector afterwards (reusing the sortPattern storage to keep track of the index permutation)                
                for (index_t i = 0; i < numValues; i++)
                {
                    //There is no more values in the matrix
                    if (searchIxSpVec >= nnz)
                        break;

                    index_t currIx = indexList4Eigen[sortPattern[i]];

                    searchInnerIndex = spMatAsVector.innerIndexPtr()[searchIxSpVec];
                    while (searchInnerIndex < currIx)
                    {
                        searchIxSpVec++;
                        searchInnerIndex = spMatAsVector.innerIndexPtr()[searchIxSpVec];                        
                    } 

                    if (searchInnerIndex == currIx)
                        triplets.emplace_back(sortPattern[i],0,spMatAsVector.valuePtr()[searchIxSpVec]);
                }

                subSpMat = std::make_shared<spMat_t>(numValues,1);
                subSpMat->setFromTriplets(triplets.begin(),triplets.end());
                
                //There's two other ways I consider doing this elegantly
                //1) Map a Sparse Vector a' over the existing values, create a slicing matrix R (similar to row/colon indexing, we don't need Q) and perform R*a';
                //   Indexing by Matrix Multiplication as found here: https://people.eecs.berkeley.edu/~aydin/spgemm_sisc12.pdf
                //2) Convert the linear index list into subscripts, perform row colon indexing, and then reshape the matrix to a vector (should be less efficient?)
                //I tried implementing 1), but it throughs bad_alloc in the matrix product

                
                //Variant 1) - gives bad_alloc in some cases for now
                //Note that index lists are actually doubles in matlab, so we cast to actuall indices
                //Temporary Sparse Vector

                //mexPrintf("We have arbitrary indices!\n");
                
                //Build the R matrix  
                
                //spMat_t R(numValues,this->getRows()*this->getCols()); //outerIndexVector may become to large in csc storage
                
                //We explicitly create a csr_matrix here to avoid overflowing the outerIndexVector
                /*
                Eigen::SparseMatrix<float,Eigen::RowMajor,index_t> R(numValues,this->getRows()*this->getCols());
                //mexPrintf("%dx%d Matrix initialized!",R.rows(),R.cols());
                R.reserve(numValues);    
                //mexPrintf("Reserved Storage!");      
                
                #pragma omp parallel for schedule(static)
                for (index_t i = 0; i < numValues; i++)
                {
                    R.valuePtr()[i] = 1;
                    R.innerIndexPtr()[i] = indexList4Eigen[i];
                    R.outerIndexPtr()[i] = i;        
                }
                R.outerIndexPtr()[numValues] = numValues;
                    
                //mexPrintf("Matrix created!");

                //We don't need Q as it would be a scalar 1


                subSpMat = std::make_shared<spMat_t>(numValues,1);
                //Perform the slicing product
                //This product may throw bad alloc when we have very large matrices. I don't know why.
                (*subSpMat) = R*spMatAsVector;
                */
            }                 
        }
        result = new sparseSingle(subSpMat);  

        if (!isColumnVector)   
            result->transposed = true;
    }
    else{
        throw(MexException("sparseSingle:invalidIndex","Unsupported index type!"));
    }

    return result;

}

sparseSingle::index_t sparseSingle::toLinearIndex(const sparseSingle::index_t row, const sparseSingle::index_t col) const
{
    return this->getRows()*col + row;
}

sparseSingle::index_t sparseSingle::linearIndexToColIndex(const sparseSingle::index_t linIx) const
{
    return linIx / this->getRows();        
}

sparseSingle::index_t sparseSingle::linearIndexToRowIndex(const sparseSingle::index_t linIx) const
{
    return linIx % this->getRows();        
}

mxArray* sparseSingle::full() const 
{
    mxArray* fullMatrix = mxCreateNumericMatrix(this->getRows(),this->getCols(),mxSINGLE_CLASS,mxREAL);
    mxSingle* fullMatrix_data = mxGetSingles(fullMatrix);

    Eigen::Map<mxSingleAsMatrix_t> fullMatrixMap(fullMatrix_data,this->getRows(),this->getCols());
    
    if (this->transposed)
        fullMatrixMap = this->eigSpMatrix->transpose().toDense();
    else
        fullMatrixMap = this->eigSpMatrix->toDense();

    return fullMatrix;
}

mxArray* sparseSingle::addDense(const mxArray* denseMx) const
{
    mxClassID mxType = mxGetClassID(denseMx);
    mwSize m = mxGetM(denseMx);
    mwSize n = mxGetN(denseMx);
    
    bool isScalar = (m == 1) & (n == 1);
    if (mxType != mxSINGLE_CLASS && !isScalar)
    {
        throw(MexException("sparseSingle:wrongDataType","Matrix addition only implemented for single/double!"));
    }
    
    bool sizeMatch = (m == this->getRows()) & (n == this->getCols());
    
    if (!isScalar && !sizeMatch)
        throw(MexException("sparseSingle:wrongOperandSize","Matrix addition only implemented for scalars and same shape! Implicit expansion not yet supported!"));

    mxArray* resultMatrix = mxCreateNumericMatrix(this->getRows(),this->getCols(),mxSINGLE_CLASS,mxREAL);
    mxSingle* resultMatrix_data = mxGetSingles(resultMatrix);
    Eigen::Map<mxSingleAsMatrix_t> resultMatrixMap(resultMatrix_data,this->getRows(),this->getCols());

    if (isScalar)
    {
        double addScalar = mxGetScalar(denseMx);        
        if (this->transposed)
            resultMatrixMap = this->eigSpMatrix->transpose().toDense();
        else
            resultMatrixMap = this->eigSpMatrix->toDense();

        resultMatrixMap.array() += (float) addScalar;
    }
    else if (sizeMatch)
    {        
        mxSingle* addMatrix_data = mxGetSingles(denseMx);
        Eigen::Map<mxSingleAsMatrix_t> addMatrixMap(addMatrix_data,m,n);
        
        //According to the Eigen documentation, first coying the dense matrix and then using += sparse is faster
        resultMatrixMap = addMatrixMap;

        if (this->transposed)
            resultMatrixMap += this->eigSpMatrix->transpose();
        else
            resultMatrixMap += *this->eigSpMatrix;
    }

    return resultMatrix;
}
//// Linear Algebra ////

sparseSingle* sparseSingle::transpose() const {
    sparseSingle* transposedCopy = new sparseSingle();
    transposedCopy->transposed = !this->transposed;
    transposedCopy->eigSpMatrix = this->eigSpMatrix;
    //this->transposed = !this->transposed;
    return transposedCopy;
}

mxArray* sparseSingle::timesVec(const mxArray* vals_) const 
{
    mwSize nCols = mxGetN(vals_); 
    mwSize nRows = mxGetM(vals_);
    if (nCols != 1 && nRows != 1)
        throw(MexException("sparseSingle:timesVec:wrongOperand","Operand is not a vector!"));
    
    sparseSingle::index_t n = mxGetNumberOfElements(vals_);
    if (n != this->getCols())
        throw(MexException("sparseSingle:timesVec:wrongSize","Operand Vector has incompatible size!"));
    
    const mxSingle* vals = mxGetSingles(vals_);

    //Create a Map to the Eigen vector
    Eigen::Map<const Eigen::VectorXf> vecMap(vals,n);
    //Create the result array and map eigen vector around it - when transposed, the getRows is already considering this
    mxArray* result = mxCreateUninitNumericMatrix(this->getRows(),1,mxSINGLE_CLASS,mxREAL);
    mxSingle* result_data = mxGetSingles(result);
    Eigen::Map<Eigen::VectorXf> resultMap(result_data,this->getRows());
    
    //Execute the product
    if (this->transposed)
        resultMap = this->eigSpMatrix->transpose()*vecMap;
    else
    {
        switch (this->cscParallelize)
        {
            case DEFAULT:
                resultMap = (*this->eigSpMatrix)*vecMap;
                break;

            case WITHIN_COLUMN:
                if (!(this-eigSpMatrix->isCompressed()))
                    throw(MexException("sparseSingle:timesVec:notCompressed","Sparse Matrix is not compressed! This should not happen..."));
                
                std::fill(std::execution::par_unseq,result_data,result_data + this->getRows(),0);

                for (index_t j = 0; j < this->eigSpMatrix->outerSize(); ++j)
                {                    
                    index_t start = this->eigSpMatrix->outerIndexPtr()[j];
                    index_t end   = this->eigSpMatrix->outerIndexPtr()[j+1];
                    index_t colNnz = end - start;

                    #pragma omp parallel for schedule(static)
                    for (index_t nzIx = start; nzIx < end; nzIx++)
                    {
                        index_t i = this->eigSpMatrix->innerIndexPtr()[nzIx];
                        float v = this->eigSpMatrix->valuePtr()[nzIx];

                        result_data[i] += v*vals[j];
                    }
                }
                break;   

            case ACROSS_COLUMN:
                if (!(this-eigSpMatrix->isCompressed()))
                    throw(MexException("sparseSingle:timesVec:notCompressed","Sparse Matrix is not compressed! This should not happen..."));

                std::fill(std::execution::par_unseq,result_data,result_data + this->getRows(),0);

                #pragma omp parallel for schedule(dynamic)
                for (index_t j = 0; j < this->eigSpMatrix->outerSize(); ++j)
                {                    
                    index_t start = this->eigSpMatrix->outerIndexPtr()[j];
                    index_t end   = this->eigSpMatrix->outerIndexPtr()[j+1];
                    index_t colNnz = end - start;

                    for (index_t nzIx = start; nzIx < end; nzIx++)
                    {
                        index_t i = this->eigSpMatrix->innerIndexPtr()[nzIx];
                        float v = this->eigSpMatrix->valuePtr()[nzIx];

                        float prod = v*vals[j];

                        #pragma omp atomic
                        result_data[i] += prod;
                    }
                }
                break;  

            default:
                throw(MexException("sparseSingle:timesVec:invalidAlgrithm","Selected parallelization algorithm not known!"));
        }
        
    }
    
    //return the bare array
    return result;
}

mxArray* sparseSingle::vecTimes(const mxArray* vals_) const 
{
    mwSize nCols = mxGetN(vals_); 
    mwSize nRows = mxGetM(vals_);
    if (nCols != 1 && nRows != 1)
        throw(MexException("sparseSingle:vecTimes:wrongOperand","Operand is not a vector!"));
    
    sparseSingle::index_t n = mxGetNumberOfElements(vals_);
    if (n != this->getRows())
        throw(MexException("sparseSingle:vecTimes:wrongSize","Operand Vector has incompatible size!"));
    
    const mxSingle* vals = mxGetSingles(vals_);

    //Create a Map to the Eigen vector
    Eigen::Map<const Eigen::VectorXf> vecMap(vals,n);

    //Create the result array and map eigen vector around it - when transposed, the getRows is already considering this
    mxArray* result = mxCreateNumericMatrix(1,this->getCols(),mxSINGLE_CLASS,mxREAL);
    mxSingle* result_data = mxGetSingles(result);
    Eigen::Map<Eigen::VectorXf> resultMap(result_data,this->getCols());
    
    //Execute the product
    if (!this->transposed)
        resultMap = this->eigSpMatrix->transpose()*vecMap;
    else
        resultMap = (*this->eigSpMatrix)*vecMap;
    
    //return the bare array
    return result;
}

sparseSingle* sparseSingle::timesScalar(const mxArray* val_) const
{
    if (!mxIsScalar(val_))
        throw(MexException("sparseSingle:mexInterface:invalidMexCall:timesScalar","Input needs to be scalar!"));
    const mxSingle* val = mxGetSingles(val_);
    mxSingle scalar = val[0];
    
    index_t numValues = this->getRows()*this->getCols();
    index_t nnz = this->getNnz();

    std::shared_ptr<spMat_t> scaledSpMat = std::make_shared<spMat_t>();
    (*scaledSpMat) = scalar*(*this->eigSpMatrix);

    //return the bare array
    sparseSingle* scaledMatrix = new sparseSingle(scaledSpMat);
    scaledMatrix->transposed = this->transposed;

    return scaledMatrix;
}