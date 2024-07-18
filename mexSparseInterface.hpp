#include "mex.h"
#include "class_handle.hpp"
#include "sparseEigen.hpp"

template<class sparseType>
void mexSparseInterface(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {	
    // Get the command string
    char cmd_[128];
	if (nrhs < 1 || mxGetString(prhs[0], cmd_, sizeof(cmd_)))
		mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall","First input should be a command string less than 128 characters long.");

    std::string cmd(cmd_);
        
    // New
    if (cmd == "new") {
        // Check parameters
        if (nlhs != 1)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:New","One output expected.");
        // Return a handle to a new C++ instance
        if (nrhs > 1 && nrhs < 8)
        {                
            try
            {
                if (nrhs == 1)
                    plhs[0] = convertPtr2Mat<sparseType>(new sparseType());
                else if (nrhs == 2)
                    plhs[0] = convertPtr2Mat<sparseType>(new sparseType(prhs[1]));
                else if (nrhs == 3)
                    plhs[0] = convertPtr2Mat<sparseType>(new sparseType(prhs[1],prhs[2]));
                else if (nrhs == 4)
                    plhs[0] = convertPtr2Mat<sparseType>(new sparseType(prhs[1],prhs[2],prhs[3]));
                else if (nrhs == 6)
                    plhs[0] = convertPtr2Mat<sparseType>(new sparseType(prhs[1],prhs[2],prhs[3],prhs[4],prhs[5]));
                else if (nrhs == 7)
                    plhs[0] = convertPtr2Mat<sparseType>(new sparseType(prhs[1],prhs[2],prhs[3],prhs[4],prhs[5],prhs[6]));
                else
                    throw(MexException("mexSparse:interface:invalidMexCall:New","This sanity check should never be reached!"));

            }
            catch (const MexException& e)
            {
                mexErrMsgIdAndTxt(e.id(), e.what());            
            }
            catch(...)
            {
                throw;
            }
        }
        else
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:New","Input invalid!");
        // We return now, as the object is constructed
        return;
    }    
    
    // For all other purposes, we need to pass the class handle as second argument.
    //Check there is a second input, which should be the class instance handle
    if (nrhs < 2)
		mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall","Second input should be a class instance handle.");
    
    // Get the class instance pointer from the second input
    sparseType* sparseType_instance = convertMat2Ptr<sparseType>(prhs[1]);
    
    if (cmd == "nnz")
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:nnz", "Unexpected Number of arguments!");
        plhs[0] = sparseType_instance->nnz();        
        return;
    }

    if (cmd == "size")
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:size","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseType_instance->size();
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            throw;
        }

        return;
    }

    if (cmd == "disp")
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:disp","Unexpected Number of arguments!");
        try {
            sparseType_instance->disp();        
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            throw;
        }
        return;
    }

    if (cmd == "full")
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:full","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseType_instance->full();                    
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            throw;
        }
        return;
    }

    if (cmd == "plus")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:add","Unexpected Number of arguments!");
        try {           
            plhs[0] = sparseType_instance->plus(prhs[2]);            
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:add","Addition failed for unknown reason!");
        }
        return;
    }

    if (cmd == "minusAsMinuend")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:minusAsMinuend","Unexpected Number of arguments!");
        try {           
            plhs[0] = sparseType_instance->minusAsMinuend(prhs[2]);            
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:minusAsMinuend","Subtraction failed for unknown reason!");
        }
        return;
    }

    if (cmd == "minusAsSubtrahend")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:minusAsSubtrahend","Unexpected Number of arguments!");
        try {           
            plhs[0] = sparseType_instance->minusAsSubtrahend(prhs[2]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:minusAsSubtrahend","Subtraction failed for unknown reason!");
        }
        return;
    }

    if (cmd == "rdivide")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:rdivide","Unexpected Number of arguments!");
        try {           
            plhs[0] = sparseType_instance->rdivide(prhs[2]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:rdivide","Division failed for unknown reason!");
        }
        return;
    }

    if (cmd == "ldivide")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:ldivide","Unexpected Number of arguments!");
        try {           
            plhs[0] = sparseType_instance->ldivide(prhs[2]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:ldivide","Division failed for unknown reason!");
        }
        return;
    }

    if (cmd == "times")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:times","Unexpected Number of arguments!");
        try {           
            plhs[0] = sparseType_instance->times(prhs[2]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:times","Multiplication failed for unknown reason!");
        }
        return;
    }

    if (cmd == "mtimesr")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:mtimesr","Unexpected Number of arguments!");
        try {           
            plhs[0] = sparseType_instance->mtimesr(prhs[2]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:mtimesl","Multiplication failed for unknown reason!");
        }
        return;
    }

    if (cmd == "mtimesl")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:mtimesl","Unexpected Number of arguments!");
        try {           
            plhs[0] = sparseType_instance->mtimesl(prhs[2]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:mtimesl","Multiplication failed for unknown reason!");
        }
        return;
    }

    if (cmd == "mldivide")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:mldivide","Unexpected Number of arguments!");
        try {           
            plhs[0] = sparseType_instance->mldivide(prhs[2]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:mldivide","Multiplication failed for unknown reason!");
        }
        return;
    }

    if (cmd == "timesVec")
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:timesVec","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseType_instance->timesVec(prhs[2]);             
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:timesVec","Product failed for unknown reason!");
        }
        return;
    }

    if (cmd == "vecTimes")
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:vecTimes","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseType_instance->vecTimes(prhs[2]);            
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:vecTimes","Product failed for unknown reason!");
        }
        return;
    }

    if (cmd == "timesScalar")
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:timesScalar","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseType_instance->timesScalar(prhs[2]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:timesScalar","Product with Scalar failed for unknown reason!");
        }
        return;
    }

    if (cmd == "uminus")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs > 2)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:uminus","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseType_instance->uminus();
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:uminus","Negation failed for unknown reason!");
        }
        return;
    }

    if (cmd == "transpose")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs > 2)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:transpose","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseType_instance->transpose();
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:unknownError:transpose","Transpose failed for unknown reason!");
        }
        return;
    }

    if (cmd == "subsrefRowCol")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 4)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:subsrefRowCol","Unexpected Number of arguments!");
        try {
            //mxArray* result = sparseType_instance->linearIndexing(prhs[2]);
            
            plhs[0] = sparseType_instance->rowColIndexing(prhs[2],prhs[3]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:subsrefRowCol","Indexing failed for unknown reason!");
        }
        return;
    }

    
    if (cmd == "linearIndexing")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:linearIndexing","Unexpected Number of arguments!");

        try {
            plhs[0] = sparseType_instance->linearIndexing(prhs[2]);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:linearIndexing","Indexing failed for unknown reason!");
        }
        return;
    }
    

    if (cmd == "find")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 2)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:find","Unexpected Number of arguments!");

        try {
            plhs[0]  = sparseType_instance->find();            
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:find","Nonzero lookup failed for unknown reason!");
        }
        return;
    }

    //Assignments
    if (cmd == "subsasgnRowCol")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 5)
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:subsasgnRowCol","Unexpected Number of arguments!");
        try {
            //mxArray* result = sparseType_instance->linearIndexing(prhs[2]);
            
            plhs[0] = sparseType_instance->rowColAssignment(prhs[2],prhs[3],prhs[4]);            
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall:subsasgnRowCol","Indexing failed for unknown reason!");
        }
        return;
    }
    

    // Delete
    if (cmd == "delete") {
        // Destroy the C++ object
        destroyObject<sparseType>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgTxt("mexSparse Delete: Unexpected arguments ignored.");
        return;
    }
    
    // Got here, so command not recognized
    mexErrMsgIdAndTxt("mexSparse:interface:invalidMexCall","Command not recognized.");
}