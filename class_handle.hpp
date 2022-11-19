#ifndef __CLASS_HANDLE_HPP__
#define __CLASS_HANDLE_HPP__

// This implementation of a class handle management via an uint64_t handle value was taken and modified from 
// Oliver Woodford (2022). Example MATLAB class wrapper for a C++ class 
// (https://www.mathworks.com/matlabcentral/fileexchange/38964-example-matlab-class-wrapper-for-a-c-class), 
// MATLAB Central File Exchange. 
// The MexException was added by Niklas Wahl, Copyright (c) 2022
// 
// Original Copyright Disclaimer: 
// Copyright (c) 2018, Oliver Woodford
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:

//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in
//       the documentation and/or other materials provided with the distribution

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "mex.h"
#include <stdint.h>
#include <string>
#include <cstring>
#include <typeinfo>
#include <exception>

class MexException : public std::exception
{
    public:       
        explicit MexException(const std::string& matlabErrID, const std::string& matlabErrMessage)
            : errID(matlabErrID), errMessage(matlabErrMessage) {}

        virtual ~MexException() noexcept {}

        virtual const char* what() const noexcept override
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


#define CLASS_HANDLE_SIGNATURE 0xFF00F0A5
template<class base> class class_handle
{
public:
    class_handle(base *ptr) : signature_m(CLASS_HANDLE_SIGNATURE), name_m(typeid(base).name()), ptr_m(ptr) {}
    ~class_handle() { signature_m = 0; delete ptr_m; }
    bool isValid() { return ((signature_m == CLASS_HANDLE_SIGNATURE) && !strcmp(name_m.c_str(), typeid(base).name())); }
    base* ptr() { return ptr_m; }

private:
    uint32_t signature_m;
    const std::string name_m;
    base* const ptr_m;
};

template<class base> inline mxArray *convertPtr2Mat(base *ptr)
{
    mexLock();
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new class_handle<base>(ptr));
    return out;
}

template<class base> inline class_handle<base> *convertMat2HandlePtr(const mxArray *in)
{
    if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS || mxIsComplex(in))
        throw(MexException("classHandle:invalidHandleInput","Handle input must be a real uint64 scalar."));
    class_handle<base> *ptr = reinterpret_cast<class_handle<base> *>(*((uint64_t *)mxGetData(in)));
    if (!ptr->isValid())
        throw(MexException("classHandle:invalidHandle","Handle not valid!"));
    return ptr;
}

template<class base> inline base *convertMat2Ptr(const mxArray *in)
{
    return convertMat2HandlePtr<base>(in)->ptr();
}

template<class base> inline void destroyObject(const mxArray *in)
{
    delete convertMat2HandlePtr<base>(in);
    mexUnlock();
}

#endif // __CLASS_HANDLE_HPP__
