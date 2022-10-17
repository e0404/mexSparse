#include "mex.h"
#include "C___class_interface/class_handle.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>

// The class that we are interfacing to
class sparseSingle
{
public:    

    typedef int64_t index_t;
    typedef Eigen::SparseMatrix<float,Eigen::ColMajor,index_t> spMat_t;
    typedef Eigen::SparseMatrix<float,Eigen::RowMajor,index_t> spMatTransposed_t;

    sparseSingle() { 
        this->eigSpMatrix = std::make_shared<spMat_t>();
    }

    sparseSingle(std::shared_ptr<spMat_t> eigSpMatrix_) {
        this->eigSpMatrix = eigSpMatrix_;        
    }

    sparseSingle(const mxArray *sparseDouble) {
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
        
        //Create the Eigen Sparse Matrix        
        try {        
            //this->eigSpMatrix = std::shared_ptr<spMat_t>(new spMat_t(nRows,nCols));
            this->eigSpMatrix = std::make_shared<spMat_t>(nRows,nCols);
            this->eigSpMatrix->makeCompressed();
            this->eigSpMatrix->reserve(nnz);
            std::transform(pr, pr+nnz, this->eigSpMatrix->valuePtr(), [](double d) -> float { return static_cast<float>(d);});    
            std::transform(ir, ir+nnz, this->eigSpMatrix->innerIndexPtr(), [](mwIndex i) -> index_t { return static_cast<index_t>(i);});
            std::transform(jc, jc+(nCols+1), this->eigSpMatrix->outerIndexPtr(), [](mwIndex i) -> index_t { return static_cast<index_t>(i);});
        }
        catch (...) {
            mexErrMsgIdAndTxt("MATLAB:sparseInternalOutput:invalidInputType","Eigen Map could not be constructed from sparse matrix!");
        }

        // no need to free memory because matlab should handle memory management of return values
    }
        ~sparseSingle() {
            #ifndef NDEBUG 
                mexPrintf("Calling destructor - %d single sparse matrix instances still exist!\n",this->eigSpMatrix.use_count() - 1);
            #endif
        }

        mxArray* timesVec(const mxSingle* vals,mwSize n) const {
            //Create a Map to the Eigen vector
            Eigen::Map<const Eigen::VectorXf> vecMap(vals,n);
            //Create the result array and map eigen vector around it - when transposed, the getRows is already considering this
            mxArray* result = mxCreateNumericMatrix(this->getRows(),1,mxSINGLE_CLASS,mxREAL);
            mxSingle* result_data = mxGetSingles(result);
            Eigen::Map<Eigen::VectorXf> resultMap(result_data,this->getRows());
            
            //Execute the product
            if (this->transposed)
                resultMap = this->eigSpMatrix->transpose()*vecMap;
            else
                resultMap = (*this->eigSpMatrix)*vecMap;
            
            //return the bare array
            return result;
        }

        mxArray* vecTimes(const mxSingle* vals,mwSize n) const {
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

        mwSize getNnz() const {
            return this->eigSpMatrix->nonZeros();
        }

        mwSize getCols() const {
            if (this->transposed)
                return this->eigSpMatrix->rows();
            else
                return this->eigSpMatrix->cols();
        }

        mwSize getRows() const {
            if (this->transposed)
                return this->eigSpMatrix->cols();
            else
                return this->eigSpMatrix->rows();
        }

        sparseSingle* transpose() const {
            sparseSingle* transposedCopy = new sparseSingle();
            transposedCopy->transposed = !this->transposed;
            transposedCopy->eigSpMatrix = this->eigSpMatrix;
            //this->transposed = !this->transposed;
            return transposedCopy;
        }

        /*
        mxArray* rowColIndexing(const std::vector<index_t>& rowIndex, const std::vector<index_t>& colIndex) const {
            if (rowIndex.size() != colIndex.size())
                mexErrMsgTxt("Index lists not matching in size!");

            index_t numValues = rowIndex.size();

            mxArray* result = mxCreateNumericMatrix(numValues,1,mxSINGLE_CLASS,mxREAL);
            mxSingle* result_data = mxGetSingles(result);

            std::fill(result_data,result_data + numValues,mxSingle(0.0));
            
            
            //#pragma omp parallel for schedule(dynamic)               
            for (index_t k = 0; k < this->eigSpMatrix->outerSize(); ++k) {//Columns
                //std::vector<index_t> indices;

                //First if we have indices in the column, very slow for now
                std::vector<index_t> colFound;
                for (index_t colIx_k = 0; colIx_k < colIndex.size(); ++colIx_k)
                {
                    if (colIndex[colIx_k] == k) {
                        colFound.push_back(colIx_k);
                    }                    
                }                

                //Now do the same for rows
                if (colFound.size() > 0)
                {     
                    for (index_t subRowIx_k = 0; subRowIx_k < colFound.size(); ++subRowIx_k)
                    {
                        index_t currIxPair = colFound[subRowIx_k];
                        index_t currRowIndex = rowIndex[currIxPair];
                        index_t currColIndex = colIndex[currIxPair];

                        //mexPrintf("%d ixPair, rowIndex = %d, colIndex = %d",currIxPair,currRowIndex,currColIndex);

                        for (spMat_t::InnerIterator it(*this->eigSpMatrix,k); it; ++it)
                        {
                            index_t currItRow = it.row();
                            index_t currItCol = it.col();

                            //mexPrintf("\t checking against [%d,%d]",currItRow,currItCol);

                            if (currItRow == currRowIndex && currItCol == currColIndex)
                            {
                                //mexErrMsgTxt("No Writing!");
                                result_data[currIxPair] = it.value();
                            }                            
                        }                        
                    }
                }
            }

            return result;

        }
        */

        sparseSingle* rowColIndexing(const mxArray * const rowIndex, const mxArray * const colIndex) const
        {
            //TODO: Transpose Implementation
            
            sparseSingle* indexedSubMatrix = nullptr;

            //Check if we are indexing a block
            bool consecutiveRows = false;
            bool consecutiveCols = false;

            const double * rowIndexData;
            const double * colIndexData;

            index_t nRowIndices, nColIndices; 

            
            if (mxIsDouble(rowIndex))
            {
                rowIndexData = mxGetPr(rowIndex);
                nRowIndices = mxGetNumberOfElements(rowIndex);
            }
            else
                mexErrMsgTxt("Invalid Row Index List");

            if (mxIsDouble(colIndex))
            {
                colIndexData = mxGetPr(colIndex);
                nColIndices = mxGetNumberOfElements(colIndex);
            }
            else
                mexErrMsgTxt("Invalid Column Index List");


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
                    mexErrMsgTxt("Transpose not implemented!");
                }
                else{
                    
                    index_t startRow = index_t(rowIndexData[0]) - 1;
                    index_t rows = nRowIndices;

                    index_t startCol = index_t(colIndexData[0]) - 1;
                    index_t cols = nColIndices;

                
                    auto block = this->eigSpMatrix->block(startRow,startCol,rows,cols);
                    std::shared_ptr<spMat_t> blockSpMat = std::make_shared<spMat_t>(block);
                    
                    indexedSubMatrix = new sparseSingle(blockSpMat);
                }
                
            }
            else
            {
                //Eigen Supports slicing for Dense Matrices Only, so we need to manually slice the matrix
                std::shared_ptr<spMat_t> subSpMat = std::make_shared<spMat_t>(nRowIndices,nColIndices);
                subSpMat->makeCompressed();

                if(this->transposed)
                {
                    mexErrMsgTxt("Transpose not implemented!");
                }
                else{
                    //Estimate maximum storage
                    index_t maxNnz = 0;
                    #pragma omp parallel for schedule(dynamic) reduction(+:maxNnz)
                    for (index_t colIx = 0; colIx < nColIndices; colIx++)
                    {
                        index_t col = index_t(colIndexData[colIx]) - 1;

                        index_t colNnz;
                        if (this->eigSpMatrix->isCompressed()) {
                            index_t startValIx = this->eigSpMatrix->outerIndexPtr()[col];
                            colNnz = this->eigSpMatrix->outerIndexPtr()[col+1]-startValIx;
                        }
                        else
                            colNnz = this->eigSpMatrix->innerNonZeroPtr()[col];

                        colNnz = std::min(nRowIndices,colNnz);
                      
                        maxNnz += colNnz;
                    }

                    //Debug
                    #ifndef NDEBUG
                        mexPrintf("Reserving space for %d values!\n",maxNnz);
                    #endif


                    //Reserve storage
                    subSpMat->reserve(maxNnz);

                    //Now fill sliced selected matrix, which is exteremely slow
                    //Alternative implementation allowing parallelization: Go over columns , extract sparse vectors and bild outer start indices, piece them together afterwards
                    index_t n = 0;
                    subSpMat->outerIndexPtr()[0] = 0;
                    //#pragma omp parallel for schedule(dynamic)
                    for (index_t colIx = 0; colIx < nColIndices; colIx++)
                    {                        
                        index_t col = index_t(colIndexData[colIx]) - 1;
                        
                        for (spMat_t::InnerIterator it(*this->eigSpMatrix,col); it; ++it)
                        {
                            index_t row = it.row();

                            bool foundRow = false;
                            index_t foundIx = 0;

                            while(!foundRow && foundIx < nRowIndices) {
                                if (row == (index_t(rowIndexData[foundIx]) - 1))
                                    foundRow = true;
                                foundIx++;
                            }

                            //Insert the new value
                            if (foundRow)
                            {
                                subSpMat->valuePtr()[n] = it.value();
                                subSpMat->innerIndexPtr()[n] = foundIx-1;
                                n++;
                            }
                        }

                        subSpMat->outerIndexPtr()[colIx+1] = n;  
                    }
                    
                    subSpMat->reserve(n);
                    indexedSubMatrix = new sparseSingle(subSpMat); 
                }
            }

            return indexedSubMatrix; 

            /*
            std::shared_ptr<spMat_t> subSpMat = std::make_shared<spMat_t>(numValues,1);
            subSpMat->reserve(nnz);            
            std::copy(subSpMat->valuePtr(),subSpMat->valuePtr() + nnz,this->eigSpMatrix->valuePtr());
            subSpMat->outerIndexPtr()[0] = index_t(0);
            subSpMat->outerIndexPtr()[1] = numValues;
            */
        }

        



                
        /*
        mxArray* linearIndexing(const mxArray* indexList) const {
            //First check if it is indeed an index list or a colon operator
            mxClassID ixType = mxGetClassID(indexList);

            mxArray* result;

            if (ixType == mxCHAR_CLASS)
                result = this->allValues();
            else if (ixType == mxDOUBLE_CLASS)
            {
                //Normal double indexing list
                mwSize nDim = mxGetNumberOfDimensions(indexList);
                if (nDim > 2)
                    mexErrMsgTxt("Indexing list has dimensionality bigger than 2!");
            
                const mwSize* ixDim = mxGetDimensions(indexList);

                index_t numValues;

                if (ixDim[0] == 1)
                    numValues = mxGetN(indexList);
                else if (ixDim[1] == 1)
                    numValues = mxGetM(indexList);
                else
                    mexErrMsgTxt("Only vector index lists are allowed!");


                std::vector<index_t> indexCasted;

                //Note that index lists are actually doubles in matlab, so we cast to actuall indices
                mxDouble* ixDouble = mxGetDoubles(indexList);

                std::vector<index_t> rowIndex(numValues);
                std::vector<index_t> colIndex(numValues);
                //std::transform(ixDouble, ixDouble+numValues, rowIndex.data(), [](ixDouble d) -> index_t { return this->linearIndexToRowIndex(static_cast<index_t>(d));});
                //std::transform(ixDouble, ixDouble+numValues, colIndex.data(), [](ixDouble d) -> index_t { return this->linearIndexToColIndex(static_cast<index_t>(d));});

                #pragma omp parallel for schedule(static)
                for (index_t i = 0; i < numValues; ++i)
                {
                    index_t tmp_ix = static_cast<index_t>(ixDouble[i]);
                    rowIndex[i] = this->linearIndexToRowIndex(static_cast<index_t>(tmp_ix) - 1);
                    colIndex[i] = this->linearIndexToColIndex(static_cast<index_t>(tmp_ix) - 1);
                }
                //mexPrintf("Extracted index list, first index [%d,%d]\n",rowIndex[0],colIndex[0]);
                //mexErrMsgTxt("obtained index lists");

                result = this->rowColIndexing(rowIndex,colIndex);
            }
            else{
                mexErrMsgTxt("Illegal index type!");
            }

            return result;

        }
        */

        //TODO: sparse version, as matlab expects a sparse output vector
        sparseSingle* allValues() const {

            index_t numValues = this->getRows()*this->getCols();
            index_t nnz = this->getNnz();
            //mxArray* result = mxCreateNumericMatrix(numValues,1,mxSINGLE_CLASS,mxREAL);
            //mxSingle* result_data = mxGetSingles(result);

            std::shared_ptr<spMat_t> subSpMat = std::make_shared<spMat_t>(numValues,1);
            subSpMat->reserve(nnz);            
            std::copy(subSpMat->valuePtr(),subSpMat->valuePtr() + nnz,this->eigSpMatrix->valuePtr());
            subSpMat->outerIndexPtr()[0] = index_t(0);
            subSpMat->outerIndexPtr()[1] = numValues;

            index_t count = 0;

            if (this->transposed)  
            {
                Eigen::Map<spMatTransposed_t> crs_transposed(this->getRows(),this->getCols(),this->getNnz(),this->eigSpMatrix->outerIndexPtr(),this->eigSpMatrix->innerIndexPtr(),this->eigSpMatrix->valuePtr());

                //#pragma omp parallel for schedule(dynamic)
                for (index_t k = 0; k < crs_transposed.outerSize(); ++k)
                    for (Eigen::Map<spMatTransposed_t>::InnerIterator it(crs_transposed,k); it; ++it)
                    {
                        //mexPrintf("\t(%d,%d)\t\t%g\n",it.row()+1,it.col()+1,it.value());
                        index_t linearIndex = this->toLinearIndex(it.row(),it.col());
                        subSpMat->innerIndexPtr()[count] = linearIndex;
                        count++;
                    }
            }
            else{
                #pragma omp parallel for schedule(dynamic)               
                for (index_t k = 0; k < this->eigSpMatrix->outerSize(); ++k)
                    for (spMat_t::InnerIterator it(*this->eigSpMatrix,k); it; ++it)
                    {
                        index_t linearIndex = this->toLinearIndex(it.row(),it.col());
                        subSpMat->innerIndexPtr()[count] = linearIndex;
                        count++;
                    }
            }                      

            sparseSingle* indexedMatrix = new sparseSingle(subSpMat);

            return indexedMatrix;
        }

        void disp() {
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

        
private:
    
    std::shared_ptr<spMat_t> eigSpMatrix; //The matrix is stored as shared pointer to allow management of copies, such that we sensibly use a matlab data class instead of a handle
    bool transposed = false; // Treats the matrix as transposed (we do not explicitly transpose it for performance)

    index_t toLinearIndex(const index_t row, const index_t col) const
    {
        return this->getRows()*col + row;
    }

    index_t linearIndexToColIndex(const index_t linIx) const
    {
        return linIx / this->getRows();        
    }

    index_t linearIndexToRowIndex(const index_t linIx) const
    {
        return linIx % this->getRows();        
    }

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
};

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
            mexErrMsgTxt("size: Unexpected arguments.");
        try {
            sparseSingle_instance->disp();        
        }
        catch (...)
        {
            mexErrMsgTxt("disp: Unexpected access violation.");
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

    /*
    if (!strcmp("linearIndexing",cmd))
    {
        if (nlhs < 0 || nlhs > 1 || nrhs > 3)
            mexErrMsgTxt("transpose: Unexpected arguments.");
        try {
            //mxArray* result = sparseSingle_instance->linearIndexing(prhs[2]);
            result = sparseSingle_instance->allValues()
            plhs[0] = convertPtr2Mat<sparseSingle>(result);
        }
        catch(...)
        {
            mexErrMsgTxt("timesVec: Product failed.");
        }
        return;
    }
    */


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
