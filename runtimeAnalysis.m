testCuda = false;

%Load all available DIJs
files = what('dijs');
files = files.mat;
[~,str] = cellfun(@fileparts,files,'UniformOutput',false);
str = cellfun(@(f) strsplit(f,'_'),str,'UniformOutput',false);
str = vertcat(str{:});
str = str(:,2:end);
%[~,mode,resStr,beamStr,bixWidthStr] = cellfun(@(f) strsplit(f,'_'),files,'UniformOutput',false);
runttimeTable = [];
for fileIx = 1:numel(files)
    file = fullfile('dijs',files{fileIx});
    currComb = str(fileIx,:);

    %Test functions
    fDw = @(A,w) A*w;    
    %fdGradD = @(A,dGrad) dGrad'*A;
    fdGradD = @(A,dGrad) A'*dGrad;

    load(file);        
    w = rand(dij.totalNumOfBixels,1);
    dGrad = rand(dij.doseGrid.numOfVoxels,1);
    
    fprintf('Runtime test: %s %s %s %s\n',currComb{1},currComb{2},currComb{3},currComb{4});
    
    %Matlab standard sparse double
    A = dij.physicalDose{1};
    t = timeit(@() fDw(A,w),1);
    runtimeTable(fileIx,1) = t;
    fprintf('\tMatlab sparse double d=Dw: %d s\n',t);
    
    
    t = timeit(@() fdGradD(A,dGrad),1);
    runtimeTable(fileIx,2) = t;
    fprintf('\tMatlab sparse double wGrad=dGrad''D: %d s\n',t);

    %sparseSingle
    tic;
    if testCuda
        A = SparseSingleGPU(dij.physicalDose{1});
    else
        A = SparseSingle(dij.physicalDose{1});
    end
    t = toc;
    runtimeTable(fileIx,3) = t;
    fprintf('\tCustom sparse single creation: %d s\n',t);

    t = timeit(@() fDw(A,w),1);
    runtimeTable(fileIx,4) = t;
    fprintf('\tCustom sparse single d=Dw: %d s\n',t);

    
    t = timeit(@() fdGradD(A,dGrad),1);
    fprintf('\tCustom sparse single wGrad=dGrad''D: %d s\n',t);    
    runtimeTable(fileIx,5) = t;
end
%%

varNames = {'Matlab sparse double Dw','Matlab sparse double dGrad''D','SparseSingle construct','Custom sparse single Dw','Custom sparse single dGrad''D'};
runtimeTableTmp = num2cell(runtimeTable,1);
runtimeTable = table(runtimeTableTmp{:},'VariableNames',varNames);
   
%%
figure;
nexttile;

speedup_Dw = runtimeTable.(varNames{4})./runtimeTable.(varNames{1});
avg = 1 - mean(speedup_Dw);
avg_std = std(speedup_Dw);
bar([runtimeTable.(varNames{1}),runtimeTable.(varNames{4})]); hold on;
%bar,'DisplayName',varNames{2}); 
legend(varNames{[1,4]});
title(sprintf('D*w, single speedup (%g+-%g)%%',avg*100,avg_std*100));
ylabel('runtime [s]');
nexttile;

speedup_dGradD = runtimeTable.(varNames{5})./runtimeTable.(varNames{2});
avg = 1 - mean(speedup_dGradD);
avg_std = std(speedup_dGradD);
bar([runtimeTable.(varNames{2}),runtimeTable.(varNames{5})]); hold on;
%bar,'DisplayName',varNames{2}); 
title(sprintf('dGrad''D, single speedup (%g+-%g)%%',avg*100,avg_std*100));
ylabel('runtime [s]');
legend(varNames{[2,5]});

    