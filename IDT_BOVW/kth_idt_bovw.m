% This script extracts all the IDT features for the training
% videos and quantizes these features to
% build a codebook to represent each shot from

% Run First
% readIDT.m ( element position )
% descriptor: 11:end

actions = {'Boxing', 'Clapping', 'Running', 'Walking', 'Waving'};
addpath('IDT_BOVW');
% Set basic paths:
basePath= '/Volumes/Kellan/datasets/experimentTL/kth';
matPath = '/Volumes/Kellan/MATLAB/ActionRecogTL';
addpath(matPath);

% Load all IDT HOG-HOF features corresponding to these videos
% offset = 5;
kth_train_globalSeqCount = 0; 

% Preload features if already computed
if exist(sprintf([matPath 'kth_train_IDTs.mat']), 'file')
    load(sprintf([matPath 'kth_train_IDTs.mat']));
    disp('Loading IDTs features for training set in kth done.');

else            
    kth_train_DirList = dir([basePath, '/idt/train/*.txt']);
    kth_train_IDTFeaturesArray = cell(length(kth_train_DirList),1);
    kth_train_ClassLabels = cell(length(kth_train_DirList),1);
    kth_train_SeqTotalFeatNum = cell(length(kth_train_DirList),1);
    kth_train_SeqTotalFeatCumSum = cell(length(kth_train_DirList),1);
    kth_train_overallTotalFeatNum = zeros(length(kth_train_DirList),1);

    for fn = 1:length(kth_train_DirList)
       filename = kth_train_DirList(fn).name;
       % Load all IDT HOG-HOF
       IDTFilename = [filename(1:end-4), '.txt'];
       disp(IDTFilename);
       kth_train_IDTDescriptor = readIDT([basePath,'/idt/train/',IDTFilename]);

       % Create feature cell array
       kth_train_idtFeaturesArray = cell(length(kth_train_IDTDescriptor),1);
       %classLabels = zeros(length(IDTDescriptor),1);
       kth_test_seqFeatNum = zeros(length(kth_train_IDTDescriptor),1);
       for i = 1:length(kth_train_IDTDescriptor)
           kth_train_idtFeaturesArray{i} = kth_train_IDTDescriptor(1,:);
           kth_test_seqFeatNum(i) = size(kth_train_idtFeaturesArray{i},1);
           kth_train_globalSeqCount = kth_train_globalSeqCount + size(kth_train_idtFeaturesArray{i},1);
       end
       kth_train_IDTFeaturesArray{fn} = kth_train_IDTDescriptor; 
       
       % Re-Label
       IDTFilename_split = strsplit(IDTFilename, '_');
       kth_train_ClassLabels{fn} = find(contains(actions, IDTFilename_split{1}));
       
       % For UCF-style idt filenames
       %kth_allClassLabels{fn} = find(contains(actions, IDTFilename(3:end-12)));
       
       kth_train_SeqTotalFeatNum{fn} = kth_test_seqFeatNum;
       kth_train_SeqTotalFeatCumSum{fn} = cumsum(kth_test_seqFeatNum);
       kth_train_overallTotalFeatNum(fn) = kth_train_SeqTotalFeatCumSum{fn}(end);
    end
    kth_train_overallTotalFeatCumSum = cumsum(kth_train_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of IDT features = ', int2str(kth_train_globalSeqCount)]);
    % For trajectory(30) + HOG(96) + HOF(108) + MBH(96+96)
    kth_train_FeaturesArray = zeros(kth_train_globalSeqCount, 30+96+108+96+96);    
    kth_train_FeaturesClassLabelArray = zeros(kth_train_globalSeqCount,1);

    for i = 1:length(kth_train_ClassLabels)
        [r,c]= size(kth_train_IDTFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   kth_train_FeaturesArray(1:kth_train_SeqTotalFeatCumSum{1}(1),:) = ...
                       kth_train_IDTFeaturesArray{1}(1,1:c);
                   kth_train_FeaturesClassLabelArray(1:kth_train_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(kth_train_ClassLabels{1},kth_train_SeqTotalFeatNum{1}(1),1);
               else
                   kth_train_FeaturesArray(kth_train_SeqTotalFeatCumSum{1}(j-1)+1:kth_train_SeqTotalFeatCumSum{1}(j),:) = ...
                       kth_train_IDTFeaturesArray{1}(j,1:c);
                   kth_train_FeaturesClassLabelArray(kth_train_SeqTotalFeatCumSum{1}(j-1)+1:kth_train_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(kth_train_ClassLabels{1},kth_train_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    kth_train_FeaturesArray(kth_train_overallTotalFeatCumSum(i-1)+1:...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(1),:) = ...
                        kth_train_IDTFeaturesArray{i}(1,1:c);
                    kth_train_FeaturesClassLabelArray(kth_train_overallTotalFeatCumSum(i-1)+1:...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(kth_train_ClassLabels{i},kth_train_SeqTotalFeatNum{i}(1),1);
                else
                    kth_train_FeaturesArray( ...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(j),:) = ...
                        kth_train_IDTFeaturesArray{i}(j,1:c);
                    kth_train_FeaturesClassLabelArray(...
                kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        kth_train_overallTotalFeatCumSum(i-1)+kth_train_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(kth_train_ClassLabels{i},kth_train_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
save([matPath 'kth_train_IDTs.mat'], 'kth_train_FeaturesArray', ...
     'kth_train_FeaturesClassLabelArray',...
     'kth_train_ClassLabels',...
     'kth_train_SeqTotalFeatNum',...
     'kth_train_SeqTotalFeatCumSum',...
     'kth_train_overallTotalFeatNum',...
     'kth_train_overallTotalFeatCumSum', ...
     '-v7.3');
 
disp('Saving train features Done.');
end


%% Generate codebook
numClusters = 4000;
numIter = 8;
numReps = 1;
sampleInd = 1;  % Whether to sample data points or use all datapoints

% If clusters already precomputed, just load
if exist(sprintf('kth-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('kth-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
elseif exist(sprintf('kth-tmp-stip-codebook.mat'), 'file')
        load(sprintf('kth-tmp-stip-codebook.mat'));

else
    disp('Clustering (maybe long time to be consumed) ...');
    kth_train_FeaturesClassLabelArray(kth_train_FeaturesClassLabelArray==0)=1;

    if sampleInd == 1
        % Multiplicative lagged Fibonacci generator
        s = RandStream('mlfg6331_64');
        numFeaturePointsPerClass = zeros(length(actions),1);
        
        for cat = 1:length(actions)
            numFeaturePointsPerClass(cat) = length(kth_train_FeaturesArray ...
                (kth_train_FeaturesClassLabelArray==cat));
        end
                
        % Randomly sample 10% train features
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
    % save tmp centers, membership
    save (sprintf('kth-tmp-stip-codebook.mat'), 'kth_centers', 'kth_membership');
end

%% find the membership for all the remaining IDTs
% Skip these codes if you have kth-stip-codbook.mat

if sampleInd == 1
    % Must check the number of feature array !!
    % batch size 5047274 / 11 = 458843
    batch_size = int32(length(kth_train_FeaturesArray) / 11);
    % batch_array = [1, 458844, 917687, 1376530, 1835373, 2294216, 2753059,...
      %               3211902, 3670745, 4129588, 4588431, 5047274];
    batch_array = 1:batch_size:length(kth_train_FeaturesArray); 
        
    % pre-allocate
    kth_final_membership = zeros(1,batch_array(end));
    
    for b = 1:length(batch_array)
        fprintf('Batch Index %d start\n', batch_array(b));
        tic;
        if b == 1
            trainToClustersDist = vl_alldist2(kth_train_FeaturesArray(1, :)', ...
                                              kth_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            kth_final_membership = sortedInd(:,1)';
       
        else
            trainToClustersDist = vl_alldist2(kth_train_FeaturesArray(...
                                                batch_array(b-1)+1:batch_array(b), :)', ...
                                                kth_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            kth_final_membership = [kth_final_membership(1:batch_array(b-1)) ...
                                       sortedInd(:,1)'];
        end
        fprintf('size of membership: (%d, %d)\n\n', size(kth_final_membership));
        toc;
   end
end

save(sprintf('kth-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                sampleInd,numClusters,numIter,numReps), ...
                'kth_centers', 'kth_final_membership', 'numClusters', ...
                '-v7.3');
disp('Saving final codebook done.');
%end


%% Now find the histograms for each of the videos
% fast to run
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
kth_train_finalRepresentation_nor = (kth_train_finalRepresentation'./ ...
                                        repmat(sum(kth_train_finalRepresentation'),numClusters,1))';

disp('Successfully Building BoVW in kth using training set!')


%% Now for the test data (get BoVW representations for test set)
% Preload features if already computed
if exist(sprintf([matPath 'kth_test_IDTs.mat']), 'file')
    load(sprintf([matPath 'kth_test_IDTs.mat']));
    disp('Loading IDTs features for testing set in kth ...');
    
else
    % Load IDT features corresponding to these videos
    %offset = 5;
    kth_test_globalSeqCount = 0; 

    kth_test_DirList = dir([basePath,'/idt/test/*.txt']);

    kth_test_IDTFeaturesArray = cell(length(kth_test_DirList),1);
    kth_test_ClassLabels = cell(length(kth_test_DirList),1);
    kth_test_SeqTotalFeatNum = cell(length(kth_test_DirList),1);
    kth_test_SeqTotalFeatCumSum = cell(length(kth_test_DirList),1);
    kth_test_overallTotalFeatNum = zeros(length(kth_test_DirList),1);

    for fn = 1:length(kth_test_DirList)
       filename = kth_test_DirList(fn).name;
       % Load all IDT for these
       IDTFilename = [filename(1:end-4), '.txt'];
       disp(IDTFilename);
       kth_test_IDTDescriptor = readIDT([basePath,'/idt/test/',IDTFilename]);

       % Now keep only those features that are in the shot boundaries ignoring
       % the first and last offset frames of each shot
       % Create feature cell array
       kth_test_idtFeaturesArray = cell(length(kth_test_IDTDescriptor),1);       
       kth_test_seqFeatNum = zeros(length(kth_test_IDTDescriptor),1);
       
       for i = 1:length(kth_test_IDTDescriptor)
           kth_test_idtFeaturesArray{i} = kth_test_IDTDescriptor(1,:);
           kth_test_seqFeatNum(i) = size(kth_test_idtFeaturesArray{i},1);
           kth_test_globalSeqCount = kth_test_globalSeqCount + size(kth_test_idtFeaturesArray{i},1);
       end
       
       kth_test_IDTFeaturesArray{fn} = kth_test_IDTDescriptor;
       % Re-Label (assign digital number to each file)
       IDTFilename_split = strsplit(IDTFilename, '_');
       kth_test_ClassLabels{fn} = find(contains(actions, IDTFilename_split{1}));
       
       kth_test_SeqTotalFeatNum{fn} = kth_test_seqFeatNum;
       kth_test_SeqTotalFeatCumSum{fn} = cumsum(kth_test_seqFeatNum);
       kth_test_overallTotalFeatNum(fn) = kth_test_SeqTotalFeatCumSum{fn}(end);
    end
    kth_test_overallTotalFeatCumSum = cumsum(kth_test_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of IDT features for test set = ', int2str(kth_test_globalSeqCount)]);
    % For trajectory(30) + HOG(96) + HOF(108) + MBH(96+96)
    kth_test_FeaturesArray = zeros(kth_test_globalSeqCount, 30+96+108+96+96);
    kth_test_FeaturesClassLabelArray = zeros(kth_test_globalSeqCount,1);
    
    for i = 1:length(kth_test_ClassLabels)
        [r,c]= size(kth_test_IDTFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   kth_test_FeaturesArray(1:kth_test_SeqTotalFeatCumSum{1}(1),:) = ...
                       kth_test_IDTFeaturesArray{1}(1,1:c);
                   kth_test_FeaturesClassLabelArray(1:kth_test_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(kth_test_ClassLabels{1},kth_test_SeqTotalFeatNum{1}(1),1);
               else
                   kth_test_FeaturesArray(kth_test_SeqTotalFeatCumSum{1}(j-1)+1:kth_test_SeqTotalFeatCumSum{1}(j),:) = ...
                       kth_test_IDTFeaturesArray{1}(j,1:c);
                   kth_test_FeaturesClassLabelArray(kth_test_SeqTotalFeatCumSum{1}(j-1)+1:kth_test_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(kth_test_ClassLabels{1},kth_test_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    kth_test_FeaturesArray(kth_test_overallTotalFeatCumSum(i-1)+1:...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(1),:) = ...
                        kth_test_IDTFeaturesArray{i}(1,1:c);
                    kth_test_FeaturesClassLabelArray(kth_test_overallTotalFeatCumSum(i-1)+1:...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(kth_test_ClassLabels{i},kth_test_SeqTotalFeatNum{i}(1),1);
                else
                    kth_test_FeaturesArray( ...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(j),:) = ...
                        kth_test_IDTFeaturesArray{i}(j,1:c);
                    kth_test_FeaturesClassLabelArray(...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        kth_test_overallTotalFeatCumSum(i-1)+kth_test_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(kth_test_ClassLabels{i},kth_test_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
    
    save([matPath 'kth_test_IDTs.mat'], ...
         'kth_test_FeaturesArray', ...
         'kth_test_FeaturesClassLabelArray',...
         'kth_test_ClassLabels',...
         'kth_test_SeqTotalFeatNum',...
         'kth_test_SeqTotalFeatCumSum',...
         'kth_test_overallTotalFeatNum',...
         'kth_test_overallTotalFeatCumSum', ...
         '-v7.3');
    disp('Saving test features Done.');

end

%% Find memberships for all these points
if exist(sprintf('kth-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('kth-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    % check total number of test feature array    
    % batch size (1658235-1) / 314 =5281

    batch_size_test = int32(length(kth_test_FeaturesArray) / 314);
    batch_array_test = 1:batch_size_test:length(kth_test_FeaturesArray); 

    % pre-allocate
    kth_test_membership = zeros(1,batch_array_test(end));

    for b = 1:length(batch_array_test)
        fprintf('Batch Index %d start\n', batch_array_test(b));
        tic;
        if b == 1
            testToClustersDist = vl_alldist2(kth_test_FeaturesArray(1, :)', ...
                                              kth_centers);
            [testToClustersDist, test_sortedInd] = sort(testToClustersDist,2);
            kth_test_membership = test_sortedInd(:,1)';

        else
            testToClustersDist = vl_alldist2(kth_test_FeaturesArray(...
                                                batch_array_test(b-1)+1:batch_array_test(b), :)', ...
                                                kth_centers);
            [testToClustersDist, test_sortedInd] = sort(testToClustersDist,2);
            kth_test_membership = [kth_test_membership(1:batch_array_test(b-1)) ...
                                       test_sortedInd(:,1)'];
        end
        fprintf('size of test membership: (%d, %d)\n\n', size(kth_test_membership));
        toc;
    end

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

    % Normalize histogram
    kth_test_finalRepresentation_nor = (kth_test_finalRepresentation'./ ...
                                            repmat(sum(kth_test_finalRepresentation'),...
                                            numClusters,1))';
    % Save train and test features
    save(sprintf('kth-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
        'kth_train_finalRepresentation','kth_train_finalRepresentation_nor',...
        'kth_test_finalRepresentation', 'kth_test_finalRepresentation_nor',...
        'kth_train_Labels', 'kth_test_Labels', 'actions');
    
    disp('Saving all BoVWs Done.');

end

%% Final Feature mat 
if exist(sprintf('kth-IDT-allFeatures-%d-numclust.mat',numClusters), 'file')
    load(sprintf('kth-IDT-allFeatures-%d-numclust.mat', numClusters));
    disp('Loading kth-IDT-allfeatures.mat file done');
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
    
    save(sprintf('kth-IDT-allFeatures-%d-numclust.mat', numClusters),...
                    'kth');  
    disp('Save all features, labels and parameters for train and test in kth')
end
disp('Everything is done !')

