if exist ('OCTAVE_VERSION','builtin')
    ccName = eval('mkoctfile -p CXX');
else
    myCXXompiler = mex.getCompilerConfigurations('C++','Selected');
    ccName = myCXXompiler.ShortName;
end

if ~isempty(strfind(ccName,'MSVC')) %Not use contains(...) because of octave
    mex -R2018a COMPFLAGS="$COMPFLAGS /openmp /std:c++17 /O2" mexSparseSingle.cpp sparseSingle.cpp -Ieigen/ -DNDEBUG
else
    mex -R2018a CXXFLAGS="-fexceptions -fno-omit-frame-pointer -fopenmp -std=gnu++17 -O3" LDFLAGS="$LDFLAGS -fopenmp" mexSparseSingle.cpp sparseSingle.cpp -Ieigen/ -DNDEBUG -v
end