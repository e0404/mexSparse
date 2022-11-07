% function to compile the cuSparse mex file for gpu acceleration
function mex_compileCUDA()
    
    setenv('MW_ALLOW_ANY_CUDA','1')
    setenv('MW_NVCC_PATH', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin\')
    
    %{
    mexcudaCmd = cell(0);
    mexcudaCmd{end+1} = '-R2018a';
    mexcudaCmd{end+1} = '-DNDEBUG';
    mexcudaCmd{end+1} = '-DCUDA_MEX_PERFANA';
    mexcudaCmd{end+1} = '-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\include"';
    mexcudaCmd{end+1} = '-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\lib\x64"';
    mexcudaCmd{end+1} = 'NVCCFLAGS=''"$NVCCFLAGS -Wno-deprecated-gpu-targets"''';
    mexcudaCmd{end+1} = 'LDFLAGS=''"$LDFLAGS -Wl,--no-as-needed"''';
    mexcudaCmd{end+1} = '-lcusparse';
    %mexcudaCmd{end+1} = '-g';
    %mexcudaCmd{end+1} = '-v';

    mexcudaCmd{end+1} = 'mexcudaSparseSingleGPU.cu';
    mexcudaCmd{end+1} = 'sparseSingleGPU.cu';
    %}
    
    try
        
        mexcuda  -R2018a mexcudaSparseSingleGPU.cu sparseSingleGPU.cu ...
            -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\include" ...
            -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\lib\x64" ...
            NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"'...
            -lcusparse -DNDEBUG -DCUDA_MEX_PERFANA  %-g -v         
        %mexcuda(mexcudaCmd{:})
    catch ME
        rethrow(ME);
    end
end