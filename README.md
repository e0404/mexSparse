Single Precision Sparse Matrix class implementation interfacing the Eigen3 C++ Template Library for linear algebra: https://eigen.tuxfamily.org. 
Developed at the German Cancer Research Center (DKFZ) by Jeremias Kunz & Niklas Wahl (Department of Medical Physics in Radiation Oncology, Group Radiotherapy Optimization)

# General Information
This "SparseSingle" implementation using Matlab's mex interface is mainly built for matRad (https://github.com/e0404/matRad) to save memory / speed-up optimization for radiotherapy. It is thus very limited in functionality, but can maybe help other developers to get started.

The project is supported by Mathworks as part of their Community Toolbox Support Program.

## CPU Implementation
The CPU implementation contains mainly basic linear algebra operations, particularly A\*x and A'\*x. 
Despite using a regular "data class" (i.e., not a handle), it is quite efficient:
- It does not need to reallocate for the transpose, this is managed using a flag. 
- The indexing operations can surpass Matlab's double sparse indexing for larger matrices.
- Parallelism is used in non-LA operations with stl's (>=c++17) execution policies and OpenMP

# GPU Implementation

There also exists a SparseSingleGPU prototype which, for now, only can do A\*x and A'\*x. It is currently implemented in a way that the Parallel Computing Toolbox is not required, by always copying input / results to / from the GPU such that Matlab only sees the array in RAM. Future work will also include returning / accepting gpuArrays, but our vision is to allow usage without the Parallel Computing Toolbox albeit reduced performance in this case. 

# Contribute
## Rules
If you want to contribute, the best thing is to overload a so far non-existing function you know / want to use from Matlab. Here's some rules:
- Functions should always be first implemented in SparseSingle before their counterpart is implemented in SparseSingleGPU
- Add a test in SparseSingleTest if you add something to the mex interface

## TODOs
### General TODOs:
- Template the class to work towards a general sparseTyped<index_t,value_t> that allows other value types and indexing types.
- Unify the interfaces for sparseSingle and sparseSingleGPU. We should build a SuperClass that defines the interface and allows multiple implementations beyond the Eigen and CUDA gpu we started here.
- Enable in-place modifications. Currently, we always need to create a new instance if we modify. Modifications are not easy due to the way Matlab treats data classes, i.e., there is no sensible way to create a copy constructor, and thus we can not check if a data object has been copied. 

### Specific TODOs for SparseSingle:
- Simplify the mex calling interface (maybe store function signatures in a map or something like that) or use switch instead of if(strcmp...)
- Solvers
- Thorough performance analysis on matrices of different shape/size/sparsity pattern compared to Matlab's sparse double

### Additional bigger TODOs for SparseSingleGPU:
- Enable use of gpuArray's: Write functions that return mxGPUArray instead of mxArray (and introduce a define that handles if we compile with or without it) to continue working with obtained results
- Evaluate performance 64 bit indexing (currently we use 32 bit)
- Add Tests

