#include "mex.h"
#include "C___class_interface/class_handle.hpp"
#include "sparseSingle.hpp"

#include <stdexcept>

// The class that we are interfacing to


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
    // Get the command string
    char cmd_[128];
	if (nrhs < 1 || mxGetString(prhs[0], cmd_, sizeof(cmd_)))
		mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall","First input should be a command string less than 128 characters long.");

    std::string cmd(cmd_);
        
    // New
    if (cmd == "new") {
        // Check parameters
        if (nlhs != 1)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:New","One output expected.");
        // Return a handle to a new C++ instance
        if (nrhs > 1 && nrhs < 4)
        {                
            try
            {
                if (nrhs == 1)
                    plhs[0] = convertPtr2Mat<sparseSingle>(new sparseSingle());
                else if (nrhs == 2)
                    plhs[0] = convertPtr2Mat<sparseSingle>(new sparseSingle(prhs[1]));
                else if (nrhs == 3)
                    plhs[0] = convertPtr2Mat<sparseSingle>(new sparseSingle(prhs[1],prhs[2]));
                else
                    throw(MexException("sparseSingle:mexInterface:invalidMexCall:New","This sanity check should never be reached!"));

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
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:New","Input invalid!");
        // We return now, as the object is constructed
        return;
    }    
    
    // For all other purposes, we need to pass the class handle as second argument.
    //Check there is a second input, which should be the class instance handle
    if (nrhs < 2)
		mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall","Second input should be a class instance handle.");
    
    // Get the class instance pointer from the second input
    sparseSingle* sparseSingle_instance = convertMat2Ptr<sparseSingle>(prhs[1]);
    
    if (cmd == "nnz")
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:nnz", "Unexpected Number of arguments!");
        plhs[0] = sparseSingle_instance->nnz();        
        return;
    }

    if (cmd == "size")
    {
        // Check parameters
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:size","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseSingle_instance->size();
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
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:disp","Unexpected Number of arguments!");
        try {
            sparseSingle_instance->disp();        
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
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:full","Unexpected Number of arguments!");
        try {
            mxArray* result = sparseSingle_instance->full();
            plhs[0] = result;       
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

    if (cmd == "addDense")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:addDense","Unexpected Number of arguments!");
        try {           
            mxArray* result = sparseSingle_instance->addDense(prhs[2]);
            plhs[0] = result;
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:unknownError:addDense","Addition failed for unknown reason!");
        }
        return;
    }

    if (cmd == "timesVec")
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:timesVec","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseSingle_instance->timesVec(prhs[2]);             
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:unknownError:timesVec","Product failed for unknown reason!");
        }
        return;
    }

    if (cmd == "vecTimes")
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:vecTimes","Unexpected Number of arguments!");
        try {
            plhs[0] = sparseSingle_instance->vecTimes(prhs[2]);            
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:unknownError:vecTimes","Product failed for unknown reason!");
        }
        return;
    }

    if (cmd == "timesScalar")
    {
        if (nlhs < 0 || nlhs > 1)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:timesScalar","Unexpected Number of arguments!");
        try {
            sparseSingle* result = sparseSingle_instance->timesScalar(prhs[2]);
            plhs[0] = convertPtr2Mat<sparseSingle>(result);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:unknownError:timesScalar","Product with Scalar failed for unknown reason!");
        }
        return;
    }

    if (cmd == "transpose")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs > 2)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:transpose","Unexpected Number of arguments!");
        try {
            plhs[0] = convertPtr2Mat<sparseSingle>(sparseSingle_instance->transpose());
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:unknownError:transpose","Transpose failed for unknown reason!");
        }
        return;
    }

    if (cmd == "subsrefRowCol")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 4)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:subsrefRowCol","Unexpected Number of arguments!");
        try {
            //mxArray* result = sparseSingle_instance->linearIndexing(prhs[2]);
            
            sparseSingle* result = sparseSingle_instance->rowColIndexing(prhs[2],prhs[3]);
            plhs[0] = convertPtr2Mat<sparseSingle>(result);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:subsrefRowCol","Indexing failed for unknown reason!");
        }
        return;
    }

    
    if (cmd == "linearIndexing")
    {
        if (nlhs < 0 || nlhs > 1 || nrhs != 3)
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:linearIndexing","Unexpected Number of arguments!");

        try {
            sparseSingle* result = sparseSingle_instance->linearIndexing(prhs[2]);
            //result = sparseSingle_instance->allValues()
            plhs[0] = convertPtr2Mat<sparseSingle>(result);
        }
        catch (const MexException& e)
        {
            mexErrMsgIdAndTxt(e.id(), e.what());            
        }
        catch(...)
        {
            mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall:linearIndexing","Indexing failed for unknown reason!");
        }
        return;
    }
    


    // Delete
    if (cmd == "delete") {
        // Destroy the C++ object
        destroyObject<sparseSingle>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgTxt("sparseSingle Delete: Unexpected arguments ignored.");
        return;
    }
    
    // Got here, so command not recognized
    mexErrMsgIdAndTxt("sparseSingle:mexInterface:invalidMexCall","Command not recognized.");
}
