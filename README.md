Single Precision Sparse Matrix class implementation interfacing the Eigen3 C++ Template Library for linear algebra: https://eigen.tuxfamily.org
Developed at the German Cancer Research Center (DKFZ) by Jeremias Kunz & Niklas Wahl (Department of Medical Physics in Radiation Oncology, Group Radiotherapy Optimization)

This "SparseSingle" implementation was mainly build for matRad (https://github.com/e0404/matRad) to save memory / speed-up optimization for radiotherapy and supported by Mathworks.
It thus contains mainly basic linear algebra operations, particularly A\*x and A'\*x. 
Despite using a regular "data class" (i.e., not a handle), it is quite efficient:
- It does not need to reallocate for the transpose, this is managed using a flag. 
- The indexing operations can surpass Matlab's double sparse indexing for larger matrices.
- Parallelism is used in non-LA operations with stl's (>=c++17) execution policies and OpenMP

There also exists a SparseSingleGPU prototype which, for now, only can do A\*x and A'\*x. It is currently implemented in a way that the Parallel Computing Toolbox is not required, by always copying input / results to / from the GPU such that Matlab only sees the array in RAM. Future work will also include returning / accepting gpuArrays, but our vision is to allow usage without the Parallel Computing Toolbox albeit reduced performance in this case. 

If you want to contribute, the best thing is to overload a so far non-existing function you know / want to use from Matlab. Here's some rules:
- Functions should always be first implemented in SparseSingle before their counterpart is implemented in SparseSingleGPU
- Add a test in SparseSingleTest if you add something to the mex interface

Bigger TODOs for SparseSingle:
- Simplify the mex calling interface (maybe store function signatures in a map or something like that) or use switch instead of if(strcmp...)
- Template the class (towards a general SparseOther that allows other data- and storage types)
- Enable modification of the class (inserts, horz-/vertcat, etc)
- Solvers
- Thorough performance analysis on matrices of different shape/size/sparsity pattern compared to Matlab's sparse double

Additional bigger TODOs for SparseSingleGPU:
- Write functions that return mxGPUArray instead of mxArray (and introduce a define that handles if we compile with or without it) to continue working with obtained results
- Evaluate performance 64 bit indexing (currently we use 32 bit)
- Add Tests

