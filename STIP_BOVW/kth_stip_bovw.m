% This script extracts all the STIP HOG-HOF features for the training
% videos according to the annotated shots and quantizes these features to
% build a codebook to represent each shot from

% Run First
% readHOGHOF_ucf.m ( element position )
% location: 5,6,7
% sigma2: 8
% tau2: 9
% descriptor: 10:end

%check every time
actions = {'Boxing', 'Clapping', 'Running', 'Walking', 'Waving'};
addpath('STIP_BOVW');
addpath('/Volumes/Kellan/datasets/experimentTL');

% Set basic paths:
basePath= '/Volumes/Kellan/datasets/experimentTL/kth' ;

% Load all STIP HOG-HOF features corresponding to these videos
% offset = 5;
kth_train_globalSeqCount = 0; 

% Preload features if already computed
if exist(sprintf('kth_train_STIPs.mat'), 'file')
    disp('Loading STIPs features for training set in KTH ...');
    load(sprintf('kth_train_STIPs.mat'));
else            
    kth_train_DirList = dir([basePath, '/stip/train/*.txt']);
    kth_train_STIPFeaturesArray = cell(length(kth_train_DirList),1);
    kth_train_ClassLabels = cell(length(kth_train_DirList),1);
    kth_train_SeqTotalFeatNum = cell(length(kth_train_DirList),1);
    kth_train_SeqTotalFeatCumSum = cell(length(kth_train_DirList),1);
    kth_train_overallTotalFeatNum = zeros(length(kth_train_DirList),1);

    for fn = 1:length(kth_train_DirList)
       filename = kth_train_DirList(fn).name;
       % Load all STIP HOG-HOF
       STIPFilename = [filename(1:end-4), '.txt'];
       disp(STIPFilename);
       [kth_train_STIPLocation, kth_train_STIPSigma2, ...
           kth_train_STIPTau2, kth_train_STIPDescriptor] = ...
          readHOGHOF_ucf([basePath,'/stip/train/', STIPFilename]);

       % Create feature cell array
       kth_train_stipFeaturesArray = cell(length(kth_train_STIPDescriptor),1);
       %classLabels = zeros(length(STIPDescriptor),1);
       kth_test_seqFeatNum = zeros(length(kth_train_STIPDescriptor),1);
       for i = 1:length(kth_train_STIPDescriptor)
           kth_train_stipFeaturesArray{i} = kth_train_STIPDescriptor(1,:);
           kth_test_seqFeatNum(i) = size(kth_train_stipFeaturesArray{i},1);
           kth_train_globalSeqCount = kth_train_globalSeqCount + size(kth_train_stipFeaturesArray{i},1);
       end
       kth_train_STIPFeaturesArray{fn} = kth_train_STIPDescriptor; 
       
       % Re-Label (check every time)
       STIPFilename_split = strsplit(STIPFilename, '_');
       kth_train_ClassLabels{fn} = find(contains(actions, STIPFilename_split{1}));
       
       % For UCF-style stip filenames
       %kth_allClassLabels{fn} = find(contains(actions, STIPFilename(3:end-12)));
       
       kth_train_SeqTotalFeatNum{fn} = kth_test_seqFeatNum;
       kth_train_SeqTotalFeatCumSum{fn} = cumsum(kth_test_seqFeatNum);
       kth_train_overallTotalFeatNum(fn) = kth_train_SeqTotalFeatCumSum{fn}(end);
    end
    kth_train_overallTotalFeatCumSum = cumsum(kth_train_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of HOG/HOF features = ', int2str(kth_train_globalSeqCount)]);
    kth_train_FeaturesArray = zeros(kth_train_globalSeqCount, 72+90);    % For HOG(72)+HOF(90)
    kth_train_FeaturesClassLabelArray = zeros(kth_train_globalSeqCount,1);

    for i = 1:length(kth_train_ClassLabels)
        [r,c]= size(kth_train_STIPFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   kth_train_FeaturesArray(1:kth_train_SeqTotalFeatCumSum{1}(1),:) = ...
                       kth_train_STIPFeaturesArray{1}(1,1:c);
                   kth_train_FeaturesClassLabelArray(1:kth_train_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(kth_train_ClassLabels{1},kth_train_SeqTotalFeatNum{1}(1),1);
               else
                   kth_train_FeaturesArray(kth_train_SeqTotalFeatCumSum{1}(j-1)+1:kth_train_SeqTotalFeatCumSum{1}(j),:) = ...
                       kth_train_STIPFeaturesArray{1}(j,1:c);
                   kth_train_FeaturesClassLabelArray(kth_train_SeqTotalFeatCumSum{1}(j-1)+1:kth_train_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(kth_train_ClassLabels{1},kth_train_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    kth_train_FeaturesArray(kth_train_overallTotalFeatCumSum(i-1)+1:...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(1),:) = ...
                        kth_train_STIPFeaturesArray{i}(1,1:c);
                    kth_train_FeaturesClassLabelArray(kth_train_overallTotalFeatCumSum(i-1)+1:...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(kth_train_ClassLabels{i},kth_train_SeqTotalFeatNum{i}(1),1);
                else
                    kth_train_FeaturesArray( ...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(j),:) = ...
                        kth_train_STIPFeaturesArray{i}(j,1:c);
                    kth_train_FeaturesClassLabelArray(...
                kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(kth_train_ClassLabels{i},kth_train_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
save('kth_train_STIPs.mat', 'kth_train_FeaturesArray', ...
     'kth_train_FeaturesClassLabelArray',...
     'kth_train_ClassLabels',...
     'kth_train_SeqTotalFeatNum',...
     'kth_train_SeqTotalFeatCumSum',...
     'kth_train_overallTotalFeatNum',...
     'kth_train_overallTotalFeatCumSum', ...
     '-v7.3');
end
%% Generate codebook
numClusters = 4000;
numIter = 8;
numReps = 1;
sampleInd = 0;  % Whether to sample data points or use all datapoints

% If clusters already precomputed, just load
if exist(sprintf('kth-STIP-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('kth-STIP-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    disp('Clustering ...');
    kth_train_FeaturesClassLabelArray(kth_train_FeaturesClassLabelArray==0)=1;

    if sampleInd == 1
        s = RandStream('mlfg6331_64');
        numFeaturePointsPerClass = zeros(length(actions),1);
        
        for cat = 1:length(actions)
            numFeaturePointsPerClass(cat) = length(kth_train_FeaturesArray ...
                (kth_train_FeaturesClassLabelArray==cat));
        end
                
        [randomSampleFeaturesPerClass, index] = datasample(s, kth_train_FeaturesArray, ...
                                int32(length(kth_train_FeaturesArray)*0.1),...
                                'Replace',false);
        randomSampleLabelsPerClass = kth_train_FeaturesClassLabelArray(index);

        kth_train_Features = randomSampleFeaturesPerClass;
        kth_train_FeaturesLabels = randomSampleLabelsPerClass;

        
    else
        % Cluster all the training features into clusters
        % If you want to cluster all the features
        kth_train_Features = kth_train_FeaturesArray;
        kth_train_FeaturesLabels = kth_train_FeaturesClassLabelArray;
    end
    
    tic;
    
    [kth_centers,kth_membership] = vl_kmeans(kth_train_Features', numClusters, 'verbose',...
                                    'algorithm', 'elkan', 'initialization', 'plusplus',...
                                    'maxnumiterations',numIter, 'numrepetitions', numReps);
    toc;
    
   
    if sampleInd == 1
        % If a subsampled training was used for clustering, 
        %find the membership for all the remaining STIPs
        if any(size(kth_train_Features) ~= size(kth_train_FeaturesArray))
            % Compute all pair distances with the cluster centers
            trainToClustersDist = vl_alldist2(kth_train_FeaturesArray', kth_centers);
            % Sort all the distances in ascending order
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);

            kth_membership = sortedInd(:,1);
        end
    end
  

    save(sprintf('kth-STIP-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
                    'kth_train_Features', 'kth_train_FeaturesLabels',...
                    'kth_centers', 'kth_membership', 'numClusters', ...
                    '-v7.3');
end


%% Now find the histograms for each of the videos
kth_train_VidNum = zeros(length(kth_train_ClassLabels),1);
for i = 1:length(kth_train_ClassLabels)
   kth_train_VidNum(i) = length(kth_train_ClassLabels{i});
end
kth_train_VidNumCumSum = cumsum(kth_train_VidNum);
kth_train_totalSeq = kth_train_VidNumCumSum(end);
kth_train_finalRepresentation = zeros(kth_train_totalSeq, numClusters);
kth_train_Labels = zeros(kth_train_totalSeq,1);

for i = 1:length(kth_train_ClassLabels)
    if i == 1
        kth_train_finalRepresentation(1,:) =  vl_ikmeanshist(numClusters,kth_membership(...
                   1:length(kth_train_SeqTotalFeatCumSum{1})));
        kth_train_Labels(i) = kth_train_ClassLabels{i};
    else
        kth_train_finalRepresentation(i,:) =  vl_ikmeanshist(numClusters,kth_membership(...
            kth_train_overallTotalFeatCumSum(i-1)+1 : kth_train_overallTotalFeatCumSum(i)));
        kth_train_Labels(i) = kth_train_ClassLabels{i};
    end
end

% Normalize histogram
kth_train_finalRepresentation_nor = (kth_train_finalRepresentation'./repmat(sum(kth_train_finalRepresentation'),numClusters,1))';

disp('Successfully Building BoVW in kth using training set!')

%% Now for the test data (get BoVW representations for test set)

% Preload features if already computed
if exist(sprintf('kth_test_STIPs.mat'), 'file')
    disp('Loading STIPs features for testing set in KTH ...');
    load(sprintf('kth_test_STIPs.mat'));
else
    % Load STIP HOG-HOF features corresponding to these videos
    %offset = 5;
    kth_test_globalSeqCount = 0; 

    kth_test_DirList = dir([basePath, '/stip/test/*.txt']);

    kth_test_STIPFeaturesArray = cell(length(kth_test_DirList),1);
    kth_test_ClassLabels = cell(length(kth_test_DirList),1);
    kth_test_SeqTotalFeatNum = cell(length(kth_test_DirList),1);
    kth_test_SeqTotalFeatCumSum = cell(length(kth_test_DirList),1);
    kth_test_overallTotalFeatNum = zeros(length(kth_test_DirList),1);

    for fn = 1:length(kth_test_DirList)
       filename = kth_test_DirList(fn).name;
       % Load all STIP HOG-HOF for these
       STIPFilename = [filename(1:end-4), '.txt'];
       disp(STIPFilename);
       [kth_test_STIPLocation, kth_test_STIPSigma2,...
           kth_test_STIPTau2, kth_test_STIPDescriptor] = ...
           readHOGHOF_ucf([basePath, '/stip/test/', STIPFilename]);

       % Now keep only those features that are in the shot boundaries ignoring
       % the first and last offset frames of each shot
       % Create feature cell array
       kth_test_stipFeaturesArray = cell(length(kth_test_STIPDescriptor),1);       
       kth_test_seqFeatNum = zeros(length(kth_test_STIPDescriptor),1);
       
       for i = 1:length(kth_test_STIPDescriptor)
           kth_test_stipFeaturesArray{i} = kth_test_STIPDescriptor(1,:);
           kth_test_seqFeatNum(i) = size(kth_test_stipFeaturesArray{i},1);
           kth_test_globalSeqCount = kth_test_globalSeqCount + size(kth_test_stipFeaturesArray{i},1);
       end
       
       kth_test_STIPFeaturesArray{fn} = kth_test_STIPDescriptor;
       % Re-Label (assign digital number to each file)
       STIPFilename_split = strsplit(STIPFilename, '_');
       kth_test_ClassLabels{fn} = find(contains(actions, STIPFilename_split{1}));
       
       kth_test_SeqTotalFeatNum{fn} = kth_test_seqFeatNum;
       kth_test_SeqTotalFeatCumSum{fn} = cumsum(kth_test_seqFeatNum);
       kth_test_overallTotalFeatNum(fn) = kth_test_SeqTotalFeatCumSum{fn}(end);
    end
    kth_test_overallTotalFeatCumSum = cumsum(kth_test_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of HOG-HOF features for test set = ', int2str(kth_test_globalSeqCount)]);
    kth_test_FeaturesArray = zeros(kth_test_globalSeqCount, 72+90);    % For HOG(72)+HOF(90)
    kth_test_FeaturesClassLabelArray = zeros(kth_test_globalSeqCount,1);
    
    for i = 1:length(kth_test_ClassLabels)
        [r,c]= size(kth_test_STIPFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   kth_test_FeaturesArray(1:kth_test_SeqTotalFeatCumSum{1}(1),:) = ...
                       kth_test_STIPFeaturesArray{1}(1,1:c);
                   kth_test_FeaturesClassLabelArray(1:kth_test_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(kth_test_ClassLabels{1},kth_test_SeqTotalFeatNum{1}(1),1);
               else
                   kth_test_FeaturesArray(kth_test_SeqTotalFeatCumSum{1}(j-1)+1:kth_test_SeqTotalFeatCumSum{1}(j),:) = ...
                       kth_test_STIPFeaturesArray{1}(j,1:c);
                   kth_test_FeaturesClassLabelArray(kth_test_SeqTotalFeatCumSum{1}(j-1)+1:kth_test_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(kth_test_ClassLabels{1},kth_test_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    kth_test_FeaturesArray(kth_test_overallTotalFeatCumSum(i-1)+1:...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(1),:) = ...
                        kth_test_STIPFeaturesArray{i}(1,1:c);
                    kth_test_FeaturesClassLabelArray(kth_test_overallTotalFeatCumSum(i-1)+1:...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(kth_test_ClassLabels{i},kth_test_SeqTotalFeatNum{i}(1),1);
                else
                    kth_test_FeaturesArray( ...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(j),:) = ...
                        kth_test_STIPFeaturesArray{i}(j,1:c);
                    kth_test_FeaturesClassLabelArray(...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(kth_test_ClassLabels{i},kth_test_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
    
    save('kth_test_STIPs.mat', ...
         'kth_test_FeaturesArray', ...
         'kth_test_FeaturesClassLabelArray',...
         'kth_test_ClassLabels',...
         'kth_test_SeqTotalFeatNum',...
         'kth_test_SeqTotalFeatCumSum',...
         'kth_test_overallTotalFeatNum',...
         'kth_test_overallTotalFeatCumSum', ...
         '-v7.3');
end


% Find memberships for all these points
if exist(sprintf('kth-STIP-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('kth-STIP-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    % Compute all pair distances with the cluster centers
    kth_testToClustersDist = vl_alldist2(kth_test_FeaturesArray', kth_centers);
    % Sort all the distances in ascending order
    [kth_testToClustersDist, sortedInd] = sort(kth_testToClustersDist,2);

    kth_test_FeaturesMembership = sortedInd(:,1);

    % Now find the histograms for each of the test videos
    % Now find the histograms for each of the videos
    kth_test_allVidNum = zeros(length(kth_test_ClassLabels),1);
    for i = 1:length(kth_test_ClassLabels)
       kth_test_allVidNum(i) = length(kth_test_ClassLabels{i});
    end
    kth_test_VidNumCumSum = cumsum(kth_test_allVidNum);
    kth_test_totalTestSeq = kth_test_VidNumCumSum(end);
    kth_test_finalRepresentation = zeros(kth_test_totalTestSeq, numClusters);
    kth_test_Labels = zeros(kth_test_totalTestSeq,1);


    for i = 1:length(kth_test_ClassLabels)
        if i == 1
            kth_test_finalRepresentation(1,:) =  vl_ikmeanshist(numClusters,kth_test_FeaturesMembership(...
                       1:length(kth_test_SeqTotalFeatCumSum{1})));
            kth_test_Labels(i) = kth_test_ClassLabels{i};
        else
            kth_test_finalRepresentation(i,:) =  vl_ikmeanshist(numClusters,kth_test_FeaturesMembership(...
                kth_test_overallTotalFeatCumSum(i-1)+1 : kth_test_overallTotalFeatCumSum(i)));
            kth_test_Labels(i) = kth_test_ClassLabels{i};
        end
    end

    % Some error checking
    %testHistSums = sum(testHists);
    %testHistSums(testHistSums == 0) = 1;

    % Normalize histogram
    kth_test_finalRepresentation_nor = (kth_test_finalRepresentation'./ ...
                                            repmat(sum(kth_test_finalRepresentation'),...
                                            numClusters,1))';
                                        
    % Save train and test features
    save(sprintf('kth-STIP-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
        'kth_train_finalRepresentation','kth_train_finalRepresentation_nor',...
        'kth_test_finalRepresentation', 'kth_test_finalRepresentation_nor',...
        'kth_train_Labels', 'kth_test_Labels', 'actions');
end

%% Final Feature mat 

if exist(sprintf('kth-STIP-allFeatures-%d-numclust.mat', numClusters))
    disp('Loading kth-STIP-allfeatures.mat file ...');
    load(sprintf('kth-STIP-allFeatures-%d-numclust.mat', numClusters));
else
    
    kth.train.features = kth_train_finalRepresentation;
    kth.test.features = kth_test_finalRepresentation;
    kth.train.normalised_features = kth_train_finalRepresentation_nor;
    kth.test.normalised_features = kth_test_finalRepresentation_nor;
    kth.train.lables = kth_train_Labels;
    kth.test.lables = kth_test_Labels;
    
    kth.bovw.numClusters = numClusters;
    kth.bovw.sampleInd = sampleInd;
    kth.bovw.numIter = numIter;
    kth.bovw.numReps = numReps; 
    
    kth.bovw.actions = actions;
    
    save(sprintf('kth-STIP-allFeatures-%d-numclust.mat', numClusters),...
                    'kth');   
end

disp('Save all features, labels and parameters for train and test in kth')
