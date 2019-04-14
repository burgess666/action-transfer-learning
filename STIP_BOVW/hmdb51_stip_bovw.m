% This script extracts all the STIP HOG-HOF features for the training
% videos according to the annotated shots and quantizes these features to
% build a codebook to represent each shot from

% Run First
% readHOGHOF_hdmb.m ( element position )
% location: 5,6,7
% sigma2: 8
% tau2: 9
% descriptor: 10:end

%check every time
actions = {'Biking', 'Clapping', 'Diving', 'GolfSwing', 'Jumping', ...
              'Punch', 'PushUps', 'Running', 'Walking', 'Waving'};

addpath('STIP_BOVW');
addpath('utils');
addpath('/Volumes/Kellan/datasets/experimentTL');

% Set basic paths:
basePath= '/Volumes/Kellan/datasets/experimentTL/hmdb51';

% Load all STIP HOG-HOF features corresponding to these videos
% offset = 5;
hmdb51_train_globalSeqCount = 0; 

% Preload features if already computed
if exist(sprintf('hmdb51_train_STIPs.mat'), 'file')
    disp('Loading STIPs features for training set in HMDB51 ...');
    load(sprintf('hmdb51_train_STIPs.mat'));
else    
    %source_DirList = dir([featuresPath, '/', char(chosenActionNames{cat}) ,'/*.txt']);
        
    hmdb51_train_DirList = dir([basePath, '/stip/train/*.txt']);
    hmdb51_train_STIPFeaturesArray = cell(length(hmdb51_train_DirList),1);
    hmdb51_train_ClassLabels = cell(length(hmdb51_train_DirList),1);
    hmdb51_train_SeqTotalFeatNum = cell(length(hmdb51_train_DirList),1);
    hmdb51_train_SeqTotalFeatCumSum = cell(length(hmdb51_train_DirList),1);
    hmdb51_train_overallTotalFeatNum = zeros(length(hmdb51_train_DirList),1);

    for fn = 1:length(hmdb51_train_DirList)
       filename = hmdb51_train_DirList(fn).name;
       % Load all STIP HOG-HOF
       STIPFilename = [filename(1:end-4), '.txt'];
       disp(STIPFilename);
       [hmdb51_train_STIPLocation, hmdb51_train_STIPSigma2, ...
           hmdb51_train_STIPTau2, hmdb51_train_STIPDescriptor] = ...
          readHOGHOF_hdmb([basePath,'/stip/train/', STIPFilename]);

       % Create feature cell array
       hmdb51_train_stipFeaturesArray = cell(length(hmdb51_train_STIPDescriptor),1);
       %classLabels = zeros(length(STIPDescriptor),1);
       hmdb51_test_seqFeatNum = zeros(length(hmdb51_train_STIPDescriptor),1);
       for i = 1:length(hmdb51_train_STIPDescriptor)
           hmdb51_train_stipFeaturesArray{i} = hmdb51_train_STIPDescriptor(1,:);
           hmdb51_test_seqFeatNum(i) = size(hmdb51_train_stipFeaturesArray{i},1);
           hmdb51_train_globalSeqCount = hmdb51_train_globalSeqCount + size(hmdb51_train_stipFeaturesArray{i},1);
       end
       hmdb51_train_STIPFeaturesArray{fn} = hmdb51_train_STIPDescriptor; 
       
       % Re-Label (check every time)
       STIPFilename_split = strsplit(STIPFilename, '_');
       hmdb51_train_ClassLabels{fn} = find(contains(actions, STIPFilename_split{1}));
       
       % For UCF-style stip filenames
       %hmdb51_allClassLabels{fn} = find(contains(actions, STIPFilename(3:end-12)));
       
       hmdb51_train_SeqTotalFeatNum{fn} = hmdb51_test_seqFeatNum;
       hmdb51_train_SeqTotalFeatCumSum{fn} = cumsum(hmdb51_test_seqFeatNum);
       hmdb51_train_overallTotalFeatNum(fn) = hmdb51_train_SeqTotalFeatCumSum{fn}(end);
    end
    hmdb51_train_overallTotalFeatCumSum = cumsum(hmdb51_train_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of HOG/HOF features = ', int2str(hmdb51_train_globalSeqCount)]);
    hmdb51_train_FeaturesArray = zeros(hmdb51_train_globalSeqCount, 72+90);    % For HOG(72)+HOF(90)
    hmdb51_train_FeaturesClassLabelArray = zeros(hmdb51_train_globalSeqCount,1);

    for i = 1:length(hmdb51_train_ClassLabels)
        [r,c]= size(hmdb51_train_STIPFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   hmdb51_train_FeaturesArray(1:hmdb51_train_SeqTotalFeatCumSum{1}(1),:) = ...
                       hmdb51_train_STIPFeaturesArray{1}(1,1:c);
                   hmdb51_train_FeaturesClassLabelArray(1:hmdb51_train_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(hmdb51_train_ClassLabels{1},hmdb51_train_SeqTotalFeatNum{1}(1),1);
               else
                   hmdb51_train_FeaturesArray(hmdb51_train_SeqTotalFeatCumSum{1}(j-1)+1:hmdb51_train_SeqTotalFeatCumSum{1}(j),:) = ...
                       hmdb51_train_STIPFeaturesArray{1}(j,1:c);
                   hmdb51_train_FeaturesClassLabelArray(hmdb51_train_SeqTotalFeatCumSum{1}(j-1)+1:hmdb51_train_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(hmdb51_train_ClassLabels{1},hmdb51_train_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    hmdb51_train_FeaturesArray(hmdb51_train_overallTotalFeatCumSum(i-1)+1:...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(1),:) = ...
                        hmdb51_train_STIPFeaturesArray{i}(1,1:c);
                    hmdb51_train_FeaturesClassLabelArray(hmdb51_train_overallTotalFeatCumSum(i-1)+1:...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(hmdb51_train_ClassLabels{i},hmdb51_train_SeqTotalFeatNum{i}(1),1);
                else
                    hmdb51_train_FeaturesArray( ...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(j),:) = ...
                        hmdb51_train_STIPFeaturesArray{i}(j,1:c);
                    hmdb51_train_FeaturesClassLabelArray(...
                hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(hmdb51_train_ClassLabels{i},hmdb51_train_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
save('hmdb51_train_STIPs.mat', 'hmdb51_train_FeaturesArray', ...
     'hmdb51_train_FeaturesClassLabelArray',...
     'hmdb51_train_ClassLabels',...
     'hmdb51_train_SeqTotalFeatNum',...
     'hmdb51_train_SeqTotalFeatCumSum',...
     'hmdb51_train_overallTotalFeatNum',...
     'hmdb51_train_overallTotalFeatCumSum', ...
     '-v7.3');
disp('Saving train features Done.'); 
end
%% Generate codebook
numClusters = 4000;
numIter = 8;
numReps = 1;
sampleInd = 0;  % Whether to sample data points or use all datapoints

% If clusters already precomputed, just load
if exist(sprintf('hmdb51-STIP-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('hmdb51-STIP-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    disp('Clustering ...');
    hmdb51_train_FeaturesClassLabelArray(hmdb51_train_FeaturesClassLabelArray==0)=1;

    if sampleInd == 1
        s = RandStream('mlfg6331_64');
        numFeaturePointsPerClass = zeros(length(actions),1);
        
        for cat = 1:length(actions)
            numFeaturePointsPerClass(cat) = length(hmdb51_train_FeaturesArray ...
                (hmdb51_train_FeaturesClassLabelArray==cat));
        end
                
        [randomSampleFeaturesPerClass, index] = datasample(s, hmdb51_train_FeaturesArray, ...
                                int32(length(hmdb51_train_FeaturesArray)*0.1),...
                                'Replace',false);
        randomSampleLabelsPerClass = hmdb51_train_FeaturesClassLabelArray(index);

        hmdb51_train_Features = randomSampleFeaturesPerClass;
        hmdb51_train_FeaturesLabels = randomSampleLabelsPerClass;

        
    else
        % Cluster all the training features into clusters
        % If you want to cluster all the features
        hmdb51_train_Features = hmdb51_train_FeaturesArray;
        hmdb51_train_FeaturesLabels = hmdb51_train_FeaturesClassLabelArray;
    end
    
    tic;
    
    [hmdb51_centers,hmdb51_membership] = vl_kmeans(hmdb51_train_Features', numClusters, 'verbose',...
                                    'algorithm', 'elkan', 'initialization', 'plusplus',...
                                    'maxnumiterations',numIter, 'numrepetitions', numReps);
    toc;
    
   
    if sampleInd == 1
        % If a subsampled training was used for clustering, 
        %find the membership for all the remaining STIPs
        if any(size(hmdb51_train_Features) ~= size(hmdb51_train_FeaturesArray))
            % Compute all pair distances with the cluster centers
            trainToClustersDist = vl_alldist2(hmdb51_train_FeaturesArray', hmdb51_centers);
            % Sort all the distances in ascending order
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);

            hmdb51_membership = sortedInd(:,1);
        end
    end
  

    save(sprintf('hmdb51-STIP-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
                    'hmdb51_train_Features', 'hmdb51_train_FeaturesLabels',...
                    'hmdb51_centers', 'hmdb51_membership', 'numClusters', ...
                    '-v7.3');
    disp('Saving codebook Done.');

end


%% Now find the histograms for each of the videos
hmdb51_train_VidNum = zeros(length(hmdb51_train_ClassLabels),1);
for i = 1:length(hmdb51_train_ClassLabels)
   hmdb51_train_VidNum(i) = length(hmdb51_train_ClassLabels{i});
end
hmdb51_train_VidNumCumSum = cumsum(hmdb51_train_VidNum);
hmdb51_train_totalSeq = hmdb51_train_VidNumCumSum(end);
hmdb51_train_finalRepresentation = zeros(hmdb51_train_totalSeq, numClusters);
hmdb51_train_Labels = zeros(hmdb51_train_totalSeq,1);

for i = 1:length(hmdb51_train_ClassLabels)
    if i == 1
        hmdb51_train_finalRepresentation(1,:) =  vl_ikmeanshist(numClusters,hmdb51_membership(...
                   1:length(hmdb51_train_SeqTotalFeatCumSum{1})));
        hmdb51_train_Labels(i) = hmdb51_train_ClassLabels{i};
    else
        hmdb51_train_finalRepresentation(i,:) =  vl_ikmeanshist(numClusters,hmdb51_membership(...
            hmdb51_train_overallTotalFeatCumSum(i-1)+1 : hmdb51_train_overallTotalFeatCumSum(i)));
        hmdb51_train_Labels(i) = hmdb51_train_ClassLabels{i};
    end
end

% Normalize histogram
hmdb51_train_finalRepresentation_nor = (hmdb51_train_finalRepresentation'./repmat(sum(hmdb51_train_finalRepresentation'),numClusters,1))';

disp('Successfully Building BoVW in hmdb51 using training set!')

%% Now for the test data (get BoVW representations for test set)

% Preload features if already computed
if exist(sprintf('hmdb51_test_STIPs.mat'), 'file')
    disp('Loading STIPs features for testing set in HMDB ...');
    load(sprintf('hmdb51_test_STIPs.mat'));
else
    % Load STIP HOG-HOF features corresponding to these videos
    %offset = 5;
    hmdb51_test_globalSeqCount = 0; 

    hmdb51_test_DirList = dir([basePath, '/stip/test/*.txt']);

    hmdb51_test_STIPFeaturesArray = cell(length(hmdb51_test_DirList),1);
    hmdb51_test_ClassLabels = cell(length(hmdb51_test_DirList),1);
    hmdb51_test_SeqTotalFeatNum = cell(length(hmdb51_test_DirList),1);
    hmdb51_test_SeqTotalFeatCumSum = cell(length(hmdb51_test_DirList),1);
    hmdb51_test_overallTotalFeatNum = zeros(length(hmdb51_test_DirList),1);

    for fn = 1:length(hmdb51_test_DirList)
       filename = hmdb51_test_DirList(fn).name;
       % Load all STIP HOG-HOF for these
       STIPFilename = [filename(1:end-4), '.txt'];
       disp(STIPFilename);
       [hmdb51_test_STIPLocation, hmdb51_test_STIPSigma2,...
           hmdb51_test_STIPTau2, hmdb51_test_STIPDescriptor] = ...
           readHOGHOF_hdmb([basePath, '/stip/test/', STIPFilename]);

       % Now keep only those features that are in the shot boundaries ignoring
       % the first and last offset frames of each shot
       % Create feature cell array
       hmdb51_test_stipFeaturesArray = cell(length(hmdb51_test_STIPDescriptor),1);       
       hmdb51_test_seqFeatNum = zeros(length(hmdb51_test_STIPDescriptor),1);
       
       for i = 1:length(hmdb51_test_STIPDescriptor)
           hmdb51_test_stipFeaturesArray{i} = hmdb51_test_STIPDescriptor(1,:);
           hmdb51_test_seqFeatNum(i) = size(hmdb51_test_stipFeaturesArray{i},1);
           hmdb51_test_globalSeqCount = hmdb51_test_globalSeqCount + size(hmdb51_test_stipFeaturesArray{i},1);
       end
       
       hmdb51_test_STIPFeaturesArray{fn} = hmdb51_test_STIPDescriptor;
       % Re-Label (assign digital number to each file)
       STIPFilename_split = strsplit(STIPFilename, '_');
       hmdb51_test_ClassLabels{fn} = find(contains(actions, STIPFilename_split{1}));
       
       hmdb51_test_SeqTotalFeatNum{fn} = hmdb51_test_seqFeatNum;
       hmdb51_test_SeqTotalFeatCumSum{fn} = cumsum(hmdb51_test_seqFeatNum);
       hmdb51_test_overallTotalFeatNum(fn) = hmdb51_test_SeqTotalFeatCumSum{fn}(end);
    end
    hmdb51_test_overallTotalFeatCumSum = cumsum(hmdb51_test_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of HOG-HOF features for test set = ', int2str(hmdb51_test_globalSeqCount)]);
    hmdb51_test_FeaturesArray = zeros(hmdb51_test_globalSeqCount, 72+90);    % For HOG(72)+HOF(90)
    hmdb51_test_FeaturesClassLabelArray = zeros(hmdb51_test_globalSeqCount,1);
    
    for i = 1:length(hmdb51_test_ClassLabels)
        [r,c]= size(hmdb51_test_STIPFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   hmdb51_test_FeaturesArray(1:hmdb51_test_SeqTotalFeatCumSum{1}(1),:) = ...
                       hmdb51_test_STIPFeaturesArray{1}(1,1:c);
                   hmdb51_test_FeaturesClassLabelArray(1:hmdb51_test_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(hmdb51_test_ClassLabels{1},hmdb51_test_SeqTotalFeatNum{1}(1),1);
               else
                   hmdb51_test_FeaturesArray(hmdb51_test_SeqTotalFeatCumSum{1}(j-1)+1:hmdb51_test_SeqTotalFeatCumSum{1}(j),:) = ...
                       hmdb51_test_STIPFeaturesArray{1}(j,1:c);
                   hmdb51_test_FeaturesClassLabelArray(hmdb51_test_SeqTotalFeatCumSum{1}(j-1)+1:hmdb51_test_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(hmdb51_test_ClassLabels{1},hmdb51_test_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    hmdb51_test_FeaturesArray(hmdb51_test_overallTotalFeatCumSum(i-1)+1:...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(1),:) = ...
                        hmdb51_test_STIPFeaturesArray{i}(1,1:c);
                    hmdb51_test_FeaturesClassLabelArray(hmdb51_test_overallTotalFeatCumSum(i-1)+1:...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(hmdb51_test_ClassLabels{i},hmdb51_test_SeqTotalFeatNum{i}(1),1);
                else
                    hmdb51_test_FeaturesArray( ...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(j),:) = ...
                        hmdb51_test_STIPFeaturesArray{i}(j,1:c);
                    hmdb51_test_FeaturesClassLabelArray(...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(hmdb51_test_ClassLabels{i},hmdb51_test_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
    
    save('hmdb51_test_STIPs.mat', ...
         'hmdb51_test_FeaturesArray', ...
         'hmdb51_test_FeaturesClassLabelArray',...
         'hmdb51_test_ClassLabels',...
         'hmdb51_test_SeqTotalFeatNum',...
         'hmdb51_test_SeqTotalFeatCumSum',...
         'hmdb51_test_overallTotalFeatNum',...
         'hmdb51_test_overallTotalFeatCumSum', ...
         '-v7.3');
    disp('Saving test features Done.');
end


% Find memberships for all these points
if exist(sprintf('hmdb51-STIP-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('hmdb51-STIP-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    % Compute all pair distances with the cluster centers
    hmdb51_testToClustersDist = vl_alldist2(hmdb51_test_FeaturesArray', hmdb51_centers);
    % Sort all the distances in ascending order
    [hmdb51_testToClustersDist, sortedInd] = sort(hmdb51_testToClustersDist,2);

    hmdb51_test_FeaturesMembership = sortedInd(:,1);

    % Now find the histograms for each of the test videos
    % Now find the histograms for each of the videos
    hmdb51_test_allVidNum = zeros(length(hmdb51_test_ClassLabels),1);
    for i = 1:length(hmdb51_test_ClassLabels)
       hmdb51_test_allVidNum(i) = length(hmdb51_test_ClassLabels{i});
    end
    hmdb51_test_VidNumCumSum = cumsum(hmdb51_test_allVidNum);
    hmdb51_test_totalTestSeq = hmdb51_test_VidNumCumSum(end);
    hmdb51_test_finalRepresentation = zeros(hmdb51_test_totalTestSeq, numClusters);
    hmdb51_test_Labels = zeros(hmdb51_test_totalTestSeq,1);


    for i = 1:length(hmdb51_test_ClassLabels)
        if i == 1
            hmdb51_test_finalRepresentation(1,:) =  vl_ikmeanshist(numClusters,hmdb51_test_FeaturesMembership(...
                       1:length(hmdb51_test_SeqTotalFeatCumSum{1})));
            hmdb51_test_Labels(i) = hmdb51_test_ClassLabels{i};
        else
            hmdb51_test_finalRepresentation(i,:) =  vl_ikmeanshist(numClusters,hmdb51_test_FeaturesMembership(...
                hmdb51_test_overallTotalFeatCumSum(i-1)+1 : hmdb51_test_overallTotalFeatCumSum(i)));
            hmdb51_test_Labels(i) = hmdb51_test_ClassLabels{i};
        end
    end

    % Some error checking
    %testHistSums = sum(testHists);
    %testHistSums(testHistSums == 0) = 1;

    % Normalize histogram
    hmdb51_test_finalRepresentation_nor = (hmdb51_test_finalRepresentation'./ ...
                                            repmat(sum(hmdb51_test_finalRepresentation'),...
                                            numClusters,1))';
                                        
    % Save train and test features
    save(sprintf('hmdb51-STIP-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
        'hmdb51_train_finalRepresentation','hmdb51_train_finalRepresentation_nor',...
        'hmdb51_test_finalRepresentation', 'hmdb51_test_finalRepresentation_nor',...
        'hmdb51_train_Labels', 'hmdb51_test_Labels', 'actions');
    disp('Saving all BOVWs Done.');

end

%% Final Feature mat 

if exist(sprintf('hmdb51-STIP-allFeatures-%d-numclust.mat', numClusters))
    disp('Loading hmdb51-STIP-allfeatures.mat file ...');
    load(sprintf('hmdb51-STIP-allFeatures-%d-numclust.mat', numClusters));
else
    
    hmdb51.train.features = hmdb51_train_finalRepresentation;
    hmdb51.test.features = hmdb51_test_finalRepresentation;
    hmdb51.train.normalised_features = hmdb51_train_finalRepresentation_nor;
    hmdb51.test.normalised_features = hmdb51_test_finalRepresentation_nor;
    hmdb51.train.lables = hmdb51_train_Labels;
    hmdb51.test.lables = hmdb51_test_Labels;
    
    hmdb51.bovw.numClusters = numClusters;
    hmdb51.bovw.sampleInd = sampleInd;
    hmdb51.bovw.numIter = numIter;
    hmdb51.bovw.numReps = numReps; 
    
    hmdb51.bovw.actions = actions;
    
    save(sprintf('hmdb51-STIP-allFeatures-%d-numclust.mat', numClusters),...
                    'hmdb51');
    disp('Save all features, labels and parameters for train and test in HMDB')
end

