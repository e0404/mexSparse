%Load all available DIJs
files = what('dijs');
files = files.mat;
[~,str] = cellfun(@fileparts,files,'UniformOutput',false);
str = cellfun(@(f) strsplit(f,'_'),str,'UniformOutput',false);
str = vertcat(str{:});
str = str(:,2:end);
%[~,mode,resStr,beamStr,bixWidthStr] = cellfun(@(f) strsplit(f,'_'),files,'UniformOutput',false);

for fileIx = 1:numel(files)
    file = fullfile('dijs',files{fileIx});
    currComb = str(fileIx,:);

    %Test function
    fAx = @(A,w) A*w;    

    load(file);        
    w = rand(dij.totalNumOfBixels,1);
    
    fprintf('Runtime test: %s %s %s %s\n',currComb{1},currComb{2},currComb{3},currComb{4});
    
    %Matlab standard sparse double
    A = dij.physicalDose{1};
    t = timeit(@() fAx(A,w),1);
    fprintf('\tMatlab sparse double d=Dw: %d s\n',t);
    
    runtimeTable(fileIx,1) = t;

    %sparseSingle
    tic;
    A = SparseSingle(dij.physicalDose{1});
    t = toc;
    fprintf('\tCustom sparse single creation: %d s\n',t);

    t = timeit(@() fAx(A,w),1);
    fprintf('\tCustom sparse single d=Dw: %d s\n',t);
    
    runtimeTable(fileIx,2) = t;
end
    
    