% This script extracts all the IDT features for the training
% videos and quantizes these features to
% build a codebook to represent each shot from

% Run First
% readIDT.m ( element position )
% descriptor: 11:end

actions = {'Biking', 'Clapping', 'Diving', 'GolfSwing', 'Jumping', ...
              'Punch', 'PushUps', 'Running', 'Walking', 'Waving'};
addpath('IDT_BOVW');
% Set basic paths:
basePath= '/Volumes/Kellan/datasets/experimentTL/hmdb51';
matPath = '/Volumes/Kellan/MATLAB/ActionRecogTL';
addpath(matPath);

% Load all IDT HOG-HOF features corresponding to these videos
% offset = 5;
hmdb51_train_globalSeqCount = 0; 

% Preload features if already computed
if exist(sprintf([matPath 'hmdb51_train_IDTs.mat']), 'file')
    load(sprintf([matPath 'hmdb51_train_IDTs.mat']);
    disp('Loading IDTs features for training set in hmdb51 done.');

else            
    hmdb51_train_DirList = dir([basePath, '/idt/train/*.txt']);
    hmdb51_train_IDTFeaturesArray = cell(length(hmdb51_train_DirList),1);
    hmdb51_train_ClassLabels = cell(length(hmdb51_train_DirList),1);
    hmdb51_train_SeqTotalFeatNum = cell(length(hmdb51_train_DirList),1);
    hmdb51_train_SeqTotalFeatCumSum = cell(length(hmdb51_train_DirList),1);
    hmdb51_train_overallTotalFeatNum = zeros(length(hmdb51_train_DirList),1);

    for fn = 1:length(hmdb51_train_DirList)
       filename = hmdb51_train_DirList(fn).name;
       % Load all IDT HOG-HOF
       IDTFilename = [filename(1:end-4), '.txt'];
       disp(IDTFilename);
       hmdb51_train_IDTDescriptor = readIDT([basePath,'/idt/train/',IDTFilename]);

       % Create feature cell array
       hmdb51_train_idtFeaturesArray = cell(length(hmdb51_train_IDTDescriptor),1);
       %classLabels = zeros(length(IDTDescriptor),1);
       hmdb51_test_seqFeatNum = zeros(length(hmdb51_train_IDTDescriptor),1);
       for i = 1:length(hmdb51_train_IDTDescriptor)
           hmdb51_train_idtFeaturesArray{i} = hmdb51_train_IDTDescriptor(1,:);
           hmdb51_test_seqFeatNum(i) = size(hmdb51_train_idtFeaturesArray{i},1);
           hmdb51_train_globalSeqCount = hmdb51_train_globalSeqCount + size(hmdb51_train_idtFeaturesArray{i},1);
       end
       hmdb51_train_IDTFeaturesArray{fn} = hmdb51_train_IDTDescriptor; 
       
       % Re-Label
       IDTFilename_split = strsplit(IDTFilename, '_');
       hmdb51_train_ClassLabels{fn} = find(contains(actions, IDTFilename_split{1}));
       
       % For UCF-style idt filenames
       %hmdb51_allClassLabels{fn} = find(contains(actions, IDTFilename(3:end-12)));
       
       hmdb51_train_SeqTotalFeatNum{fn} = hmdb51_test_seqFeatNum;
       hmdb51_train_SeqTotalFeatCumSum{fn} = cumsum(hmdb51_test_seqFeatNum);
       hmdb51_train_overallTotalFeatNum(fn) = hmdb51_train_SeqTotalFeatCumSum{fn}(end);
    end
    hmdb51_train_overallTotalFeatCumSum = cumsum(hmdb51_train_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of IDT features = ', int2str(hmdb51_train_globalSeqCount)]);
    % For trajectory(30) + HOG(96) + HOF(108) + MBH(96+96)
    hmdb51_train_FeaturesArray = zeros(hmdb51_train_globalSeqCount, 30+96+108+96+96);    
    hmdb51_train_FeaturesClassLabelArray = zeros(hmdb51_train_globalSeqCount,1);

    for i = 1:length(hmdb51_train_ClassLabels)
        [r,c]= size(hmdb51_train_IDTFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   hmdb51_train_FeaturesArray(1:hmdb51_train_SeqTotalFeatCumSum{1}(1),:) = ...
                       hmdb51_train_IDTFeaturesArray{1}(1,1:c);
                   hmdb51_train_FeaturesClassLabelArray(1:hmdb51_train_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(hmdb51_train_ClassLabels{1},hmdb51_train_SeqTotalFeatNum{1}(1),1);
               else
                   hmdb51_train_FeaturesArray(hmdb51_train_SeqTotalFeatCumSum{1}(j-1)+1:hmdb51_train_SeqTotalFeatCumSum{1}(j),:) = ...
                       hmdb51_train_IDTFeaturesArray{1}(j,1:c);
                   hmdb51_train_FeaturesClassLabelArray(hmdb51_train_SeqTotalFeatCumSum{1}(j-1)+1:hmdb51_train_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(hmdb51_train_ClassLabels{1},hmdb51_train_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    hmdb51_train_FeaturesArray(hmdb51_train_overallTotalFeatCumSum(i-1)+1:...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(1),:) = ...
                        hmdb51_train_IDTFeaturesArray{i}(1,1:c);
                    hmdb51_train_FeaturesClassLabelArray(hmdb51_train_overallTotalFeatCumSum(i-1)+1:...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(hmdb51_train_ClassLabels{i},hmdb51_train_SeqTotalFeatNum{i}(1),1);
                else
                    hmdb51_train_FeaturesArray( ...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(j),:) = ...
                        hmdb51_train_IDTFeaturesArray{i}(j,1:c);
                    hmdb51_train_FeaturesClassLabelArray(...
                hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        hmdb51_train_overallTotalFeatCumSum(i-1)+hmdb51_train_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(hmdb51_train_ClassLabels{i},hmdb51_train_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
save([matPath 'hmdb51_train_IDTs.mat'], 'hmdb51_train_FeaturesArray', ...
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
sampleInd = 1;  % Whether to sample data points or use all datapoints

% If clusters already precomputed, just load
if exist(sprintf('hmdb51-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('hmdb51-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
elseif exist(sprintf('hmdb51-tmp-stip-codebook.mat'), 'file')
        load(sprintf('hmdb51-tmp-stip-codebook.mat'));

else
    disp('Clustering (maybe long time to be consumed) ...');
    hmdb51_train_FeaturesClassLabelArray(hmdb51_train_FeaturesClassLabelArray==0)=1;

    if sampleInd == 1
        % Multiplicative lagged Fibonacci generator
        s = RandStream('mlfg6331_64');
        numFeaturePointsPerClass = zeros(length(actions),1);
        
        for cat = 1:length(actions)
            numFeaturePointsPerClass(cat) = length(hmdb51_train_FeaturesArray ...
                (hmdb51_train_FeaturesClassLabelArray==cat));
        end
                
        % Randomly sample 10% train features
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
    % save tmp centers, membership
    save (sprintf('hmdb51-tmp-stip-codebook.mat'), 'hmdb51_centers', 'hmdb51_membership');
end

%% find the membership for all the remaining IDTs
% Skip these codes if you have hmdb51-stip-codbook.mat

if sampleInd == 1
    % Must check the number of feature array !!
    % batch size 5047274 / 11 = 458843
    batch_size = int32(length(hmdb51_train_FeaturesArray) / 11);
    % batch_array = [1, 458844, 917687, 1376530, 1835373, 2294216, 2753059,...
      %               3211902, 3670745, 4129588, 4588431, 5047274];
    batch_array = 1:batch_size:length(hmdb51_train_FeaturesArray); 
        
    % pre-allocate
    hmdb51_final_membership = zeros(1,batch_array(end));
    
    for b = 1:length(batch_array)
        fprintf('Batch Index %d start\n', batch_array(b));
        tic;
        if b == 1
            trainToClustersDist = vl_alldist2(hmdb51_train_FeaturesArray(1, :)', ...
                                              hmdb51_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            hmdb51_final_membership = sortedInd(:,1)';
       
        else
            trainToClustersDist = vl_alldist2(hmdb51_train_FeaturesArray(...
                                                batch_array(b-1)+1:batch_array(b), :)', ...
                                                hmdb51_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            hmdb51_final_membership = [hmdb51_final_membership(1:batch_array(b-1)) ...
                                       sortedInd(:,1)'];
        end
        fprintf('size of membership: (%d, %d)\n\n', size(hmdb51_final_membership));
        toc;
   end
end

save(sprintf('hmdb51-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                sampleInd,numClusters,numIter,numReps), ...
                'hmdb51_centers', 'hmdb51_final_membership', 'numClusters', ...
                '-v7.3');
disp('Saving final codebook done.');
%end


%% Now find the histograms for each of the videos
% fast to run
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
hmdb51_train_finalRepresentation_nor = (hmdb51_train_finalRepresentation'./ ...
                                        repmat(sum(hmdb51_train_finalRepresentation'),numClusters,1))';

disp('Successfully Building BoVW in hmdb51 using training set!')


%% Now for the test data (get BoVW representations for test set)
% Preload features if already computed
if exist(sprintf([matPath 'hmdb51_test_IDTs.mat']), 'file')
    load(sprintf([matPath 'hmdb51_test_IDTs.mat']));
    disp('Loading IDTs features for testing set in hmdb51 ...');

else
    % Load IDT features corresponding to these videos
    %offset = 5;
    hmdb51_test_globalSeqCount = 0; 

    hmdb51_test_DirList = dir([basePath,'/idt/test/*.txt']);

    hmdb51_test_IDTFeaturesArray = cell(length(hmdb51_test_DirList),1);
    hmdb51_test_ClassLabels = cell(length(hmdb51_test_DirList),1);
    hmdb51_test_SeqTotalFeatNum = cell(length(hmdb51_test_DirList),1);
    hmdb51_test_SeqTotalFeatCumSum = cell(length(hmdb51_test_DirList),1);
    hmdb51_test_overallTotalFeatNum = zeros(length(hmdb51_test_DirList),1);

    for fn = 1:length(hmdb51_test_DirList)
       filename = hmdb51_test_DirList(fn).name;
       % Load all IDT for these
       IDTFilename = [filename(1:end-4), '.txt'];
       disp(IDTFilename);
       hmdb51_test_IDTDescriptor = readIDT([basePath,'/idt/test/',IDTFilename]);

       % Now keep only those features that are in the shot boundaries ignoring
       % the first and last offset frames of each shot
       % Create feature cell array
       hmdb51_test_idtFeaturesArray = cell(length(hmdb51_test_IDTDescriptor),1);       
       hmdb51_test_seqFeatNum = zeros(length(hmdb51_test_IDTDescriptor),1);
       
       for i = 1:length(hmdb51_test_IDTDescriptor)
           hmdb51_test_idtFeaturesArray{i} = hmdb51_test_IDTDescriptor(1,:);
           hmdb51_test_seqFeatNum(i) = size(hmdb51_test_idtFeaturesArray{i},1);
           hmdb51_test_globalSeqCount = hmdb51_test_globalSeqCount + size(hmdb51_test_idtFeaturesArray{i},1);
       end
       
       hmdb51_test_IDTFeaturesArray{fn} = hmdb51_test_IDTDescriptor;
       % Re-Label (assign digital number to each file)
       IDTFilename_split = strsplit(IDTFilename, '_');
       hmdb51_test_ClassLabels{fn} = find(contains(actions, IDTFilename_split{1}));
       
       hmdb51_test_SeqTotalFeatNum{fn} = hmdb51_test_seqFeatNum;
       hmdb51_test_SeqTotalFeatCumSum{fn} = cumsum(hmdb51_test_seqFeatNum);
       hmdb51_test_overallTotalFeatNum(fn) = hmdb51_test_SeqTotalFeatCumSum{fn}(end);
    end
    hmdb51_test_overallTotalFeatCumSum = cumsum(hmdb51_test_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of IDT features for test set = ', int2str(hmdb51_test_globalSeqCount)]);
    % For trajectory(30) + HOG(96) + HOF(108) + MBH(96+96)
    hmdb51_test_FeaturesArray = zeros(hmdb51_test_globalSeqCount, 30+96+108+96+96);
    hmdb51_test_FeaturesClassLabelArray = zeros(hmdb51_test_globalSeqCount,1);
    
    for i = 1:length(hmdb51_test_ClassLabels)
        [r,c]= size(hmdb51_test_IDTFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   hmdb51_test_FeaturesArray(1:hmdb51_test_SeqTotalFeatCumSum{1}(1),:) = ...
                       hmdb51_test_IDTFeaturesArray{1}(1,1:c);
                   hmdb51_test_FeaturesClassLabelArray(1:hmdb51_test_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(hmdb51_test_ClassLabels{1},hmdb51_test_SeqTotalFeatNum{1}(1),1);
               else
                   hmdb51_test_FeaturesArray(hmdb51_test_SeqTotalFeatCumSum{1}(j-1)+1:hmdb51_test_SeqTotalFeatCumSum{1}(j),:) = ...
                       hmdb51_test_IDTFeaturesArray{1}(j,1:c);
                   hmdb51_test_FeaturesClassLabelArray(hmdb51_test_SeqTotalFeatCumSum{1}(j-1)+1:hmdb51_test_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(hmdb51_test_ClassLabels{1},hmdb51_test_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    hmdb51_test_FeaturesArray(hmdb51_test_overallTotalFeatCumSum(i-1)+1:...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(1),:) = ...
                        hmdb51_test_IDTFeaturesArray{i}(1,1:c);
                    hmdb51_test_FeaturesClassLabelArray(hmdb51_test_overallTotalFeatCumSum(i-1)+1:...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(hmdb51_test_ClassLabels{i},hmdb51_test_SeqTotalFeatNum{i}(1),1);
                else
                    hmdb51_test_FeaturesArray( ...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(j),:) = ...
                        hmdb51_test_IDTFeaturesArray{i}(j,1:c);
                    hmdb51_test_FeaturesClassLabelArray(...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        hmdb51_test_overallTotalFeatCumSum(i-1)+hmdb51_test_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(hmdb51_test_ClassLabels{i},hmdb51_test_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
    
    save([matPath 'hmdb51_test_IDTs.mat'], ...
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

%% Find memberships for all these points
if exist(sprintf('hmdb51-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('hmdb51-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    % check total number of test feature array    
    % batch size (1658235-1) / 314 =5281

    batch_size_test = int32(length(hmdb51_test_FeaturesArray) / 314);
    batch_array_test = 1:batch_size_test:length(hmdb51_test_FeaturesArray); 

    % pre-allocate
    hmdb51_test_membership = zeros(1,batch_array_test(end));

    for b = 1:length(batch_array_test)
        fprintf('Batch Index %d start\n', batch_array_test(b));
        tic;
        if b == 1
            testToClustersDist = vl_alldist2(hmdb51_test_FeaturesArray(1, :)', ...
                                              hmdb51_centers);
            [testToClustersDist, test_sortedInd] = sort(testToClustersDist,2);
            hmdb51_test_membership = test_sortedInd(:,1)';

        else
            testToClustersDist = vl_alldist2(hmdb51_test_FeaturesArray(...
                                                batch_array_test(b-1)+1:batch_array_test(b), :)', ...
                                                hmdb51_centers);
            [testToClustersDist, test_sortedInd] = sort(testToClustersDist,2);
            hmdb51_test_membership = [hmdb51_test_membership(1:batch_array_test(b-1)) ...
                                       test_sortedInd(:,1)'];
        end
        fprintf('size of test membership: (%d, %d)\n\n', size(hmdb51_test_membership));
        toc;
    end

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

    % Normalize histogram
    hmdb51_test_finalRepresentation_nor = (hmdb51_test_finalRepresentation'./ ...
                                            repmat(sum(hmdb51_test_finalRepresentation'),...
                                            numClusters,1))';
    % Save train and test features
    save(sprintf('hmdb51-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
        'hmdb51_train_finalRepresentation','hmdb51_train_finalRepresentation_nor',...
        'hmdb51_test_finalRepresentation', 'hmdb51_test_finalRepresentation_nor',...
        'hmdb51_train_Labels', 'hmdb51_test_Labels', 'actions');
    
    disp('Saving all BoVWs Done.');

end

%% Final Feature mat 
if exist(sprintf('hmdb51-IDT-allFeatures-%d-numclust.mat',numClusters), 'file')
    load(sprintf('hmdb51-IDT-allFeatures-%d-numclust.mat', numClusters));
    disp('Loading hmdb51-IDT-allfeatures.mat file done');
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
    
    save(sprintf('hmdb51-IDT-allFeatures-%d-numclust.mat', numClusters),...
                    'hmdb51');  
    disp('Save all features, labels and parameters for train and test in hmdb51')
end
disp('Everything is done !')
