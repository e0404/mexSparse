if exist ('OCTAVE_VERSION','builtin')
    ccName = eval('mkoctfile -p CXX');
else
    myCXXompiler = mex.getCompilerConfigurations('C++','Selected');
    ccName = myCXXompiler.ShortName;
end

%Compile sparse interface with 64 bit indexing but single precision
if ~isempty(strfind(ccName,'MSVC')) %Not use contains(...) because of octave
    mex -R2018a COMPFLAGS="$COMPFLAGS /openmp /std:c++17 /O2" mexSparseSingle.cpp sparseEigen.cpp -Ieigen/ -DNDEBUG
else
    mex -R2018a CXXFLAGS="-fexceptions -fno-omit-frame-pointer -fopenmp -std=c++17 -O3" LDFLAGS="$LDFLAGS -std=c++17 -fopenmp" mexSparseSingle.cpp sparseEigen.cpp -Ieigen/ -DNDEBUG -v
end

%Compile sparse interface with 32 bit indexing but single precision
if ~isempty(strfind(ccName,'MSVC')) %Not use contains(...) because of octave
    mex -R2018a COMPFLAGS="$COMPFLAGS /openmp /std:c++17 /O2" mexSparseSingle32.cpp sparseEigen.cpp -Ieigen/ -DNDEBUG
else
    mex -R2018a CXXFLAGS="-fexceptions -fno-omit-frame-pointer -fopenmp -std=c++17 -O3" LDFLAGS="$LDFLAGS -std=c++17 -fopenmp" mexSparseSingle32.cpp sparseEigen.cpp -Ieigen/ -DNDEBUG -v
end

%Compile sparse interface with 64 bit indexing and double precision (useful
%for performance comparisons
if ~isempty(strfind(ccName,'MSVC')) %Not use contains(...) because of octave
    mex -R2018a COMPFLAGS="$COMPFLAGS /openmp /std:c++17 /O2" mexSparseDouble.cpp sparseEigen.cpp -Ieigen/ -DNDEBUG
else
    mex -R2018a CXXFLAGS="-fexceptions -fno-omit-frame-pointer -fopenmp -std=c++17 -O3" LDFLAGS="$LDFLAGS -std=c++17 -fopenmp" mexSparseSingle32.cpp sparseEigen.cpp -Ieigen/ -DNDEBUG -v
end