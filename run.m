clear;
clc;

addpath(genpath('./'));

resultdir1 = 'Results/';
if (~exist('Results', 'file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end

resultdir2 = 'aResults/';
if (~exist('aResults', 'file'))
    mkdir('aResults');
    addpath(genpath('aResults/')); 
end

datadir='.\datasets\';

dataname={'MSRCV1_3v'};

numdata = length(dataname); % number of the test datasets
numname = {'_Per0.1', '_Per0.2', '_Per0.3', '_Per0.4','_Per0.5', '_Per0.6', '_Per0.7', '_Per0.8', '_Per0.9'};

for idata =1:1:1
    ResBest = zeros(9,8);
    ResStd = zeros(9,8);
    for dataIndex = 1:1:9
        datafile = [datadir, cell2mat(dataname(idata)), cell2mat(numname(dataIndex)), '.mat'];
        load(datafile);
        %data preparation...
        gt = truelabel{1};
        numview = length(data);
        cls_num = length(unique(gt));
        k= cls_num;
        tic;
        [X1, ind] = findindex(data, index);
        time1 = toc;
        maxAcc = 0;
        TempLambda1 = [0.001 1 10];
        TempLambda2 = [0.001 1 10];
        ACC = zeros(length(TempLambda1),length(TempLambda2));
        NMI = zeros(length(TempLambda1),length(TempLambda2));
        Purity = zeros(length(TempLambda1),length(TempLambda2));
        idx = 1;
        for LambdaIndex1 = 1 : length(TempLambda1)
            lambda1 = TempLambda1(LambdaIndex1);
            for LambdaIndex2 = 1 : length(TempLambda2)
                lambda2 = TempLambda2(LambdaIndex2);
                
                disp([char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2)]);
                tic;
                ini = 2; %Initialze G: 1:K-means  2:PCA
                [S,G,iter,obj] = SCSL(X1,gt,lambda1,lambda2,ind,index,ini);   
                F = SpectralClustering((S+S')/2, cls_num);
                time2 = toc;
                stream = RandStream.getGlobalStream;
                reset(stream);
                MAXiter = 1000; % Maximum number of iterations for KMeans
                REPlic = 20; % Number of replications for KMeans
                tic;
                for rep = 1 : 20
                    pY = kmeans(F, cls_num, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
                    res(rep, : ) = Clustering8Measure(gt, pY);
                end
                time3 = toc;
                runtime(idx) = time1 + time2 + time3/20;
                disp(['runtime:', num2str(runtime(idx))])
                idx = idx + 1;
                tempResBest(dataIndex, : ) = mean(res);
                tempResStd(dataIndex, : ) = std(res);
                ACC(LambdaIndex1,LambdaIndex2) = tempResBest(dataIndex, 1);
                NMI(LambdaIndex1,LambdaIndex2) = tempResBest(dataIndex, 2);
                Purity(LambdaIndex1,LambdaIndex2) = tempResBest(dataIndex, 3);
                save([resultdir1, char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2), ...
                    '-acc=', num2str(tempResBest(dataIndex,1)), '_result.mat'], 'tempResBest', 'tempResStd');
                for tempIndex = 1 : 8
                    if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                        ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                        ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                    end
                end
            end
        end
        aRuntime = mean(runtime);
        PResBest = ResBest(dataIndex, :);
        PResStd = ResStd(dataIndex, :);
        save([resultdir2, char(dataname(idata)), char(numname(dataIndex)), 'ACC_', num2str(max(ACC(:))), '_result.mat'], 'ACC', 'NMI', 'Purity', 'aRuntime', ...
             'PResBest', 'PResStd');
    end
    save([resultdir2, char(dataname(idata)), '_result.mat'], 'ResBest', 'ResStd');
end
