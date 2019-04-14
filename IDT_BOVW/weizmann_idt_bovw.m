% This script extracts all the IDT features for the training
% videos and quantizes these features to
% build a codebook to represent each shot from

% Run First
% readIDT.m ( element position )
% descriptor: 11:end

actions = {'Jumping', 'Running', 'Walking', 'Waving'};
addpath('IDT_BOVW');
addpath('/Volumes/Kellan/datasets/experimentTL');

% Set basic paths:
basePath= '/Volumes/Kellan/datasets/experimentTL/weizmann';

% Load all IDT HOG-HOF features corresponding to these videos
% offset = 5;
weizmann_train_globalSeqCount = 0; 

% Preload features if already computed
if exist(sprintf('weizmann_train_IDTs.mat'), 'file')
    load(sprintf('weizmann_train_IDTs.mat'));
    disp('Loading IDTs features for training set in Weizmann done.');

else            
    weizmann_train_DirList = dir([basePath, '/idt/train/*.txt']);
    weizmann_train_IDTFeaturesArray = cell(length(weizmann_train_DirList),1);
    weizmann_train_ClassLabels = cell(length(weizmann_train_DirList),1);
    weizmann_train_SeqTotalFeatNum = cell(length(weizmann_train_DirList),1);
    weizmann_train_SeqTotalFeatCumSum = cell(length(weizmann_train_DirList),1);
    weizmann_train_overallTotalFeatNum = zeros(length(weizmann_train_DirList),1);

    for fn = 1:length(weizmann_train_DirList)
       filename = weizmann_train_DirList(fn).name;
       % Load all IDT HOG-HOF
       IDTFilename = [filename(1:end-4), '.txt'];
       disp(IDTFilename);
       weizmann_train_IDTDescriptor = readIDT([basePath,'/idt/train/',IDTFilename]);

       % Create feature cell array
       weizmann_train_idtFeaturesArray = cell(length(weizmann_train_IDTDescriptor),1);
       %classLabels = zeros(length(IDTDescriptor),1);
       weizmann_test_seqFeatNum = zeros(length(weizmann_train_IDTDescriptor),1);
       for i = 1:length(weizmann_train_IDTDescriptor)
           weizmann_train_idtFeaturesArray{i} = weizmann_train_IDTDescriptor(1,:);
           weizmann_test_seqFeatNum(i) = size(weizmann_train_idtFeaturesArray{i},1);
           weizmann_train_globalSeqCount = weizmann_train_globalSeqCount + size(weizmann_train_idtFeaturesArray{i},1);
       end
       weizmann_train_IDTFeaturesArray{fn} = weizmann_train_IDTDescriptor; 
       
       % Re-Label
       IDTFilename_split = strsplit(IDTFilename, '_');
       weizmann_train_ClassLabels{fn} = find(contains(actions, IDTFilename_split{1}));
       
       % For UCF-style idt filenames
       %weizmann_allClassLabels{fn} = find(contains(actions, IDTFilename(3:end-12)));
       
       weizmann_train_SeqTotalFeatNum{fn} = weizmann_test_seqFeatNum;
       weizmann_train_SeqTotalFeatCumSum{fn} = cumsum(weizmann_test_seqFeatNum);
       weizmann_train_overallTotalFeatNum(fn) = weizmann_train_SeqTotalFeatCumSum{fn}(end);
    end
    weizmann_train_overallTotalFeatCumSum = cumsum(weizmann_train_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of IDT features = ', int2str(weizmann_train_globalSeqCount)]);
    % For trajectory(30) + HOG(96) + HOF(108) + MBH(96+96)
    weizmann_train_FeaturesArray = zeros(weizmann_train_globalSeqCount, 30+96+108+96+96);    
    weizmann_train_FeaturesClassLabelArray = zeros(weizmann_train_globalSeqCount,1);

    for i = 1:length(weizmann_train_ClassLabels)
        [r,c]= size(weizmann_train_IDTFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   weizmann_train_FeaturesArray(1:weizmann_train_SeqTotalFeatCumSum{1}(1),:) = ...
                       weizmann_train_IDTFeaturesArray{1}(1,1:c);
                   weizmann_train_FeaturesClassLabelArray(1:weizmann_train_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(weizmann_train_ClassLabels{1},weizmann_train_SeqTotalFeatNum{1}(1),1);
               else
                   weizmann_train_FeaturesArray(weizmann_train_SeqTotalFeatCumSum{1}(j-1)+1:weizmann_train_SeqTotalFeatCumSum{1}(j),:) = ...
                       weizmann_train_IDTFeaturesArray{1}(j,1:c);
                   weizmann_train_FeaturesClassLabelArray(weizmann_train_SeqTotalFeatCumSum{1}(j-1)+1:weizmann_train_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(weizmann_train_ClassLabels{1},weizmann_train_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    weizmann_train_FeaturesArray(weizmann_train_overallTotalFeatCumSum(i-1)+1:...
                        weizmann_train_overallTotalFeatCumSum(i-1)+weizmann_train_SeqTotalFeatCumSum{i}(1),:) = ...
                        weizmann_train_IDTFeaturesArray{i}(1,1:c);
                    weizmann_train_FeaturesClassLabelArray(weizmann_train_overallTotalFeatCumSum(i-1)+1:...
                        weizmann_train_overallTotalFeatCumSum(i-1)+weizmann_train_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(weizmann_train_ClassLabels{i},weizmann_train_SeqTotalFeatNum{i}(1),1);
                else
                    weizmann_train_FeaturesArray( ...
                        weizmann_train_overallTotalFeatCumSum(i-1)+weizmann_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        weizmann_train_overallTotalFeatCumSum(i-1)+weizmann_train_SeqTotalFeatCumSum{i}(j),:) = ...
                        weizmann_train_IDTFeaturesArray{i}(j,1:c);
                    weizmann_train_FeaturesClassLabelArray(...
                weizmann_train_overallTotalFeatCumSum(i-1)+weizmann_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        weizmann_train_overallTotalFeatCumSum(i-1)+weizmann_train_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(weizmann_train_ClassLabels{i},weizmann_train_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
save('weizmann_train_IDTs.mat', 'weizmann_train_FeaturesArray', ...
     'weizmann_train_FeaturesClassLabelArray',...
     'weizmann_train_ClassLabels',...
     'weizmann_train_SeqTotalFeatNum',...
     'weizmann_train_SeqTotalFeatCumSum',...
     'weizmann_train_overallTotalFeatNum',...
     'weizmann_train_overallTotalFeatCumSum', ...
     '-v7.3');
end

%% Generate codebook
numClusters = 4000;
numIter = 8;
numReps = 1;
sampleInd = 0;  % Whether to sample data points or use all datapoints

% If clusters already precomputed, just load
if exist(sprintf('weizmann-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('weizmann-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    disp('Clustering ...');
    weizmann_train_FeaturesClassLabelArray(weizmann_train_FeaturesClassLabelArray==0)=1;

    if sampleInd == 1
        s = RandStream('mlfg6331_64');
        numFeaturePointsPerClass = zeros(length(actions),1);
        
        for cat = 1:length(actions)
            numFeaturePointsPerClass(cat) = length(weizmann_train_FeaturesArray ...
                (weizmann_train_FeaturesClassLabelArray==cat));
        end
                
        [randomSampleFeaturesPerClass, index] = datasample(s, weizmann_train_FeaturesArray, ...
                                int32(length(weizmann_train_FeaturesArray)*0.1),...
                                'Replace',false);
        randomSampleLabelsPerClass = weizmann_train_FeaturesClassLabelArray(index);

        weizmann_train_Features = randomSampleFeaturesPerClass;
        weizmann_train_FeaturesLabels = randomSampleLabelsPerClass;

        
    else
        % Cluster all the training features into clusters
        % If you want to cluster all the features
        weizmann_train_Features = weizmann_train_FeaturesArray;
        weizmann_train_FeaturesLabels = weizmann_train_FeaturesClassLabelArray;
    end
    
    tic;
    
    [weizmann_centers,weizmann_membership] = vl_kmeans(weizmann_train_Features', numClusters, 'verbose',...
                                    'algorithm', 'elkan', 'initialization', 'plusplus',...
                                    'maxnumiterations',numIter, 'numrepetitions', numReps);
    toc;
    
   
    if sampleInd == 1
        % If a subsampled training was used for clustering, 
        %find the membership for all the remaining IDTs
        if any(size(weizmann_train_Features) ~= size(weizmann_train_FeaturesArray))
            % Compute all pair distances with the cluster centers
            trainToClustersDist = vl_alldist2(weizmann_train_FeaturesArray', weizmann_centers);
            % Sort all the distances in ascending order
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);

            weizmann_membership = sortedInd(:,1);
        end
    end
  

    save(sprintf('weizmann-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
                    'weizmann_train_Features', 'weizmann_train_FeaturesLabels',...
                    'weizmann_centers', 'weizmann_membership', 'numClusters', ...
                    '-v7.3');
end


%% Now find the histograms for each of the videos
weizmann_train_VidNum = zeros(length(weizmann_train_ClassLabels),1);
for i = 1:length(weizmann_train_ClassLabels)
   weizmann_train_VidNum(i) = length(weizmann_train_ClassLabels{i});
end
weizmann_train_VidNumCumSum = cumsum(weizmann_train_VidNum);
weizmann_train_totalSeq = weizmann_train_VidNumCumSum(end);
weizmann_train_finalRepresentation = zeros(weizmann_train_totalSeq, numClusters);
weizmann_train_Labels = zeros(weizmann_train_totalSeq,1);

for i = 1:length(weizmann_train_ClassLabels)
    if i == 1
        weizmann_train_finalRepresentation(1,:) =  vl_ikmeanshist(numClusters,weizmann_membership(...
                   1:length(weizmann_train_SeqTotalFeatCumSum{1})));
        weizmann_train_Labels(i) = weizmann_train_ClassLabels{i};
    else
        weizmann_train_finalRepresentation(i,:) =  vl_ikmeanshist(numClusters,weizmann_membership(...
            weizmann_train_overallTotalFeatCumSum(i-1)+1 : weizmann_train_overallTotalFeatCumSum(i)));
        weizmann_train_Labels(i) = weizmann_train_ClassLabels{i};
    end
end

% Normalize histogram
weizmann_train_finalRepresentation_nor = (weizmann_train_finalRepresentation'./ ...
                                        repmat(sum(weizmann_train_finalRepresentation'),numClusters,1))';

disp('Successfully Building BoVW in weizmann using training set!')

%% Now for the test data (get BoVW representations for test set)

% Preload features if already computed
if exist(sprintf('weizmann_test_IDTs.mat'), 'file')
    disp('Loading IDTs features for testing set in Weizmann ...');
    load(sprintf('weizmann_test_IDTs.mat'));
else
    % Load IDT HOG-HOF features corresponding to these videos
    %offset = 5;
    weizmann_test_globalSeqCount = 0; 

    weizmann_test_DirList = dir([basePath,'/idt/test/*.txt']);

    weizmann_test_IDTFeaturesArray = cell(length(weizmann_test_DirList),1);
    weizmann_test_ClassLabels = cell(length(weizmann_test_DirList),1);
    weizmann_test_SeqTotalFeatNum = cell(length(weizmann_test_DirList),1);
    weizmann_test_SeqTotalFeatCumSum = cell(length(weizmann_test_DirList),1);
    weizmann_test_overallTotalFeatNum = zeros(length(weizmann_test_DirList),1);

    for fn = 1:length(weizmann_test_DirList)
       filename = weizmann_test_DirList(fn).name;
       % Load all IDT for these
       IDTFilename = [filename(1:end-4), '.txt'];
       disp(IDTFilename);
       weizmann_test_IDTDescriptor = readIDT([basePath,'/idt/test/',IDTFilename]);

       % Now keep only those features that are in the shot boundaries ignoring
       % the first and last offset frames of each shot
       % Create feature cell array
       weizmann_test_idtFeaturesArray = cell(length(weizmann_test_IDTDescriptor),1);       
       weizmann_test_seqFeatNum = zeros(length(weizmann_test_IDTDescriptor),1);
       
       for i = 1:length(weizmann_test_IDTDescriptor)
           weizmann_test_idtFeaturesArray{i} = weizmann_test_IDTDescriptor(1,:);
           weizmann_test_seqFeatNum(i) = size(weizmann_test_idtFeaturesArray{i},1);
           weizmann_test_globalSeqCount = weizmann_test_globalSeqCount + size(weizmann_test_idtFeaturesArray{i},1);
       end
       
       weizmann_test_IDTFeaturesArray{fn} = weizmann_test_IDTDescriptor;
       % Re-Label (assign digital number to each file)
       IDTFilename_split = strsplit(IDTFilename, '_');
       weizmann_test_ClassLabels{fn} = find(contains(actions, IDTFilename_split{1}));
       
       weizmann_test_SeqTotalFeatNum{fn} = weizmann_test_seqFeatNum;
       weizmann_test_SeqTotalFeatCumSum{fn} = cumsum(weizmann_test_seqFeatNum);
       weizmann_test_overallTotalFeatNum(fn) = weizmann_test_SeqTotalFeatCumSum{fn}(end);
    end
    weizmann_test_overallTotalFeatCumSum = cumsum(weizmann_test_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of IDT features for test set = ', int2str(weizmann_test_globalSeqCount)]);
    % For trajectory(30) + HOG(96) + HOF(108) + MBH(96+96)
    weizmann_test_FeaturesArray = zeros(weizmann_test_globalSeqCount, 30+96+108+96+96);
    weizmann_test_FeaturesClassLabelArray = zeros(weizmann_test_globalSeqCount,1);
    
    for i = 1:length(weizmann_test_ClassLabels)
        [r,c]= size(weizmann_test_IDTFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   weizmann_test_FeaturesArray(1:weizmann_test_SeqTotalFeatCumSum{1}(1),:) = ...
                       weizmann_test_IDTFeaturesArray{1}(1,1:c);
                   weizmann_test_FeaturesClassLabelArray(1:weizmann_test_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(weizmann_test_ClassLabels{1},weizmann_test_SeqTotalFeatNum{1}(1),1);
               else
                   weizmann_test_FeaturesArray(weizmann_test_SeqTotalFeatCumSum{1}(j-1)+1:weizmann_test_SeqTotalFeatCumSum{1}(j),:) = ...
                       weizmann_test_IDTFeaturesArray{1}(j,1:c);
                   weizmann_test_FeaturesClassLabelArray(weizmann_test_SeqTotalFeatCumSum{1}(j-1)+1:weizmann_test_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(weizmann_test_ClassLabels{1},weizmann_test_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    weizmann_test_FeaturesArray(weizmann_test_overallTotalFeatCumSum(i-1)+1:...
                        weizmann_test_overallTotalFeatCumSum(i-1)+weizmann_test_SeqTotalFeatCumSum{i}(1),:) = ...
                        weizmann_test_IDTFeaturesArray{i}(1,1:c);
                    weizmann_test_FeaturesClassLabelArray(weizmann_test_overallTotalFeatCumSum(i-1)+1:...
                        weizmann_test_overallTotalFeatCumSum(i-1)+weizmann_test_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(weizmann_test_ClassLabels{i},weizmann_test_SeqTotalFeatNum{i}(1),1);
                else
                    weizmann_test_FeaturesArray( ...
                        weizmann_test_overallTotalFeatCumSum(i-1)+weizmann_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        weizmann_test_overallTotalFeatCumSum(i-1)+weizmann_test_SeqTotalFeatCumSum{i}(j),:) = ...
                        weizmann_test_IDTFeaturesArray{i}(j,1:c);
                    weizmann_test_FeaturesClassLabelArray(...
                        weizmann_test_overallTotalFeatCumSum(i-1)+weizmann_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        weizmann_test_overallTotalFeatCumSum(i-1)+weizmann_test_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(weizmann_test_ClassLabels{i},weizmann_test_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
    
    save('weizmann_test_IDTs.mat', ...
         'weizmann_test_FeaturesArray', ...
         'weizmann_test_FeaturesClassLabelArray',...
         'weizmann_test_ClassLabels',...
         'weizmann_test_SeqTotalFeatNum',...
         'weizmann_test_SeqTotalFeatCumSum',...
         'weizmann_test_overallTotalFeatNum',...
         'weizmann_test_overallTotalFeatCumSum', ...
         '-v7.3');
end

% Find memberships for all these points
if exist(sprintf('weizmann-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('weizmann-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    % Compute all pair distances with the cluster centers
    weizmann_testToClustersDist = vl_alldist2(weizmann_test_FeaturesArray', weizmann_centers);
    % Sort all the distances in ascending order
    [weizmann_testToClustersDist, sortedInd] = sort(weizmann_testToClustersDist,2);

    weizmann_test_FeaturesMembership = sortedInd(:,1);

    % Now find the histograms for each of the test videos
    % Now find the histograms for each of the videos
    weizmann_test_allVidNum = zeros(length(weizmann_test_ClassLabels),1);
    for i = 1:length(weizmann_test_ClassLabels)
       weizmann_test_allVidNum(i) = length(weizmann_test_ClassLabels{i});
    end
    weizmann_test_VidNumCumSum = cumsum(weizmann_test_allVidNum);
    weizmann_test_totalTestSeq = weizmann_test_VidNumCumSum(end);
    weizmann_test_finalRepresentation = zeros(weizmann_test_totalTestSeq, numClusters);
    weizmann_test_Labels = zeros(weizmann_test_totalTestSeq,1);


    for i = 1:length(weizmann_test_ClassLabels)
        if i == 1
            weizmann_test_finalRepresentation(1,:) =  vl_ikmeanshist(numClusters,weizmann_test_FeaturesMembership(...
                       1:length(weizmann_test_SeqTotalFeatCumSum{1})));
            weizmann_test_Labels(i) = weizmann_test_ClassLabels{i};
        else
            weizmann_test_finalRepresentation(i,:) =  vl_ikmeanshist(numClusters,weizmann_test_FeaturesMembership(...
                weizmann_test_overallTotalFeatCumSum(i-1)+1 : weizmann_test_overallTotalFeatCumSum(i)));
            weizmann_test_Labels(i) = weizmann_test_ClassLabels{i};
        end
    end

    % Normalize histogram
    weizmann_test_finalRepresentation_nor = (weizmann_test_finalRepresentation'./ ...
                                            repmat(sum(weizmann_test_finalRepresentation'),...
                                            numClusters,1))';
    % Save train and test features
    save(sprintf('weizmann-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
        'weizmann_train_finalRepresentation','weizmann_train_finalRepresentation_nor',...
        'weizmann_test_finalRepresentation', 'weizmann_test_finalRepresentation_nor',...
        'weizmann_train_Labels', 'weizmann_test_Labels', 'actions');

end

%% Final Feature mat 

if exist(sprintf('weizmann-IDT-allFeatures-%d-numclust.mat', ...
                    numClusters), 'file')
    disp('Loading weizmann-IDT-allfeatures.mat file ...');
    load(sprintf('weizmann-IDT-allFeatures-%d-numclust.mat', numClusters));
else
    
    weizmann.train.features = weizmann_train_finalRepresentation;
    weizmann.test.features = weizmann_test_finalRepresentation;
    weizmann.train.normalised_features = weizmann_train_finalRepresentation_nor;
    weizmann.test.normalised_features = weizmann_test_finalRepresentation_nor;
    weizmann.train.lables = weizmann_train_Labels;
    weizmann.test.lables = weizmann_test_Labels;
    
    weizmann.bovw.numClusters = numClusters;
    weizmann.bovw.sampleInd = sampleInd;
    weizmann.bovw.numIter = numIter;
    weizmann.bovw.numReps = numReps; 
    
    weizmann.bovw.actions = actions;
    
    save(sprintf('weizmann-IDT-allFeatures-%d-numclust.mat', numClusters),...
                    'weizmann');    
end

disp('Save all features, labels and parameters for train and test in weizmann')
