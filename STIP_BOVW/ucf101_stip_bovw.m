% This script extracts all the STIP HOG-HOF features for the training
% videos according to the annotated shots and quantizes these features to
% build a codebook to represent each shot from

% Run First
% readHOGHOF_ucf.m ( element position )
% location: 5,6,7
% sigma2: 8
% tau2: 9
% descriptor: 10:end

% BoxingPunchingBag and BoxingSpeedBag become to Boxing category
actions = {'Biking', 'Boxing', 'Diving', 'GolfSwing', ...
                  'Jumping', 'Punch', 'PushUps', 'Walking'};
              
addpath('STIP_BOVW');
addpath('utils');
addpath('/Volumes/Kellan/datasets/experimentTL');

% Set basic paths:
basePath= '/Volumes/Kellan/datasets/experimentTL/ucf101' ;

% Load all STIP HOG-HOF features corresponding to these videos
% offset = 5;
ucf101_train_globalSeqCount = 0; 

% Preload features if already computed
if exist(sprintf('ucf101_train_STIPs.mat'), 'file')
    disp('Loading STIPs features for training set in UCF101 ...');
    load(sprintf('ucf101_train_STIPs.mat'));
else            
    ucf101_train_DirList = dir([basePath, '/stip/train/*.txt']);
    ucf101_train_STIPFeaturesArray = cell(length(ucf101_train_DirList),1);
    ucf101_train_ClassLabels = cell(length(ucf101_train_DirList),1);
    ucf101_train_SeqTotalFeatNum = cell(length(ucf101_train_DirList),1);
    ucf101_train_SeqTotalFeatCumSum = cell(length(ucf101_train_DirList),1);
    ucf101_train_overallTotalFeatNum = zeros(length(ucf101_train_DirList),1);

    for fn = 1:length(ucf101_train_DirList)
       filename = ucf101_train_DirList(fn).name;
       % Load all STIP HOG-HOF
       STIPFilename = [filename(1:end-4), '.txt'];
       disp(STIPFilename);
       [ucf101_train_STIPLocation, ucf101_train_STIPSigma2, ...
           ucf101_train_STIPTau2, ucf101_train_STIPDescriptor] = ...
          readHOGHOF_ucf([basePath,'/stip/train/', STIPFilename]);

       % Create feature cell array
       ucf101_train_stipFeaturesArray = cell(length(ucf101_train_STIPDescriptor),1);
       %classLabels = zeros(length(STIPDescriptor),1);
       ucf101_test_seqFeatNum = zeros(length(ucf101_train_STIPDescriptor),1);
       for i = 1:length(ucf101_train_STIPDescriptor)
           ucf101_train_stipFeaturesArray{i} = ucf101_train_STIPDescriptor(1,:);
           ucf101_test_seqFeatNum(i) = size(ucf101_train_stipFeaturesArray{i},1);
           ucf101_train_globalSeqCount = ucf101_train_globalSeqCount + size(ucf101_train_stipFeaturesArray{i},1);
       end
       ucf101_train_STIPFeaturesArray{fn} = ucf101_train_STIPDescriptor; 
       
       % Re-Label (check every time) 
       % UCF-style (v_BoxingPunchingBag_g01_c01.avi)
       % Combine BoxingPunchingBag and BoxingSpeedBag together
       action_in_filename = STIPFilename(3:end-12);
       if (strcmp(action_in_filename, 'BoxingPunchingBag') || ...
          strcmp(action_in_filename, 'BoxingSpeedBag'))
           ucf101_train_ClassLabels{fn} = find(contains(actions, ...
                                                action_in_filename(1:6)));
       else
           ucf101_train_ClassLabels{fn} = find(contains(actions, ...
                                                        action_in_filename));
       end

       %STIPFilename_split = strsplit(STIPFilename, '_');
       %ucf101_train_ClassLabels{fn} = find(contains(actions, STIPFilename_split(2)));
       
              
       ucf101_train_SeqTotalFeatNum{fn} = ucf101_test_seqFeatNum;
       ucf101_train_SeqTotalFeatCumSum{fn} = cumsum(ucf101_test_seqFeatNum);
       ucf101_train_overallTotalFeatNum(fn) = ucf101_train_SeqTotalFeatCumSum{fn}(end);
    end
    ucf101_train_overallTotalFeatCumSum = cumsum(ucf101_train_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of HOG/HOF features = ', int2str(ucf101_train_globalSeqCount)]);
    ucf101_train_FeaturesArray = zeros(ucf101_train_globalSeqCount, 72+90);    % For HOG(72)+HOF(90)
    ucf101_train_FeaturesClassLabelArray = zeros(ucf101_train_globalSeqCount,1);

    for i = 1:length(ucf101_train_ClassLabels)
        [r,c]= size(ucf101_train_STIPFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   ucf101_train_FeaturesArray(1:ucf101_train_SeqTotalFeatCumSum{1}(1),:) = ...
                       ucf101_train_STIPFeaturesArray{1}(1,1:c);
                   ucf101_train_FeaturesClassLabelArray(1:ucf101_train_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(ucf101_train_ClassLabels{1},ucf101_train_SeqTotalFeatNum{1}(1),1);
               else
                   ucf101_train_FeaturesArray(ucf101_train_SeqTotalFeatCumSum{1}(j-1)+1:ucf101_train_SeqTotalFeatCumSum{1}(j),:) = ...
                       ucf101_train_STIPFeaturesArray{1}(j,1:c);
                   ucf101_train_FeaturesClassLabelArray(ucf101_train_SeqTotalFeatCumSum{1}(j-1)+1:ucf101_train_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(ucf101_train_ClassLabels{1},ucf101_train_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    ucf101_train_FeaturesArray(ucf101_train_overallTotalFeatCumSum(i-1)+1:...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(1),:) = ...
                        ucf101_train_STIPFeaturesArray{i}(1,1:c);
                    ucf101_train_FeaturesClassLabelArray(ucf101_train_overallTotalFeatCumSum(i-1)+1:...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(ucf101_train_ClassLabels{i},ucf101_train_SeqTotalFeatNum{i}(1),1);
                else
                    ucf101_train_FeaturesArray( ...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j),:) = ...
                        ucf101_train_STIPFeaturesArray{i}(j,1:c);
                    ucf101_train_FeaturesClassLabelArray(...
                ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(ucf101_train_ClassLabels{i},ucf101_train_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
save('ucf101_train_STIPs.mat', 'ucf101_train_FeaturesArray', ...
     'ucf101_train_FeaturesClassLabelArray',...
     'ucf101_train_ClassLabels',...
     'ucf101_train_SeqTotalFeatNum',...
     'ucf101_train_SeqTotalFeatCumSum',...
     'ucf101_train_overallTotalFeatNum',...
     'ucf101_train_overallTotalFeatCumSum', ...
     '-v7.3');
end
%% Generate codebook
numClusters = 4000;
numIter = 8;
numReps = 1;
sampleInd = 1;  % Whether to sample data points or use all datapoints

% If clusters already precomputed, just load
if exist(sprintf('ucf101-STIP-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('ucf101-STIP-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
elseif exist(sprintf('ucf101-tmp-stip-codebook.mat'), 'file')
        load(sprintf('ucf101-tmp-stip-codebook.mat'));
    
else
    disp('Clustering ...');
    ucf101_train_FeaturesClassLabelArray(ucf101_train_FeaturesClassLabelArray==0)=1;

    if sampleInd == 1
        % Multiplicative lagged Fibonacci generator
        s = RandStream('mlfg6331_64');
        numFeaturePointsPerClass = zeros(length(actions),1);
        
        for cat = 1:length(actions)
            numFeaturePointsPerClass(cat) = length(ucf101_train_FeaturesArray ...
                (ucf101_train_FeaturesClassLabelArray==cat));
        end
        
        % Randomly sample 10% train features
        [randomSampleFeaturesPerClass, index] = datasample(s, ucf101_train_FeaturesArray, ...
                                int32(length(ucf101_train_FeaturesArray)*0.1),...
                                'Replace',false);
        randomSampleLabelsPerClass = ucf101_train_FeaturesClassLabelArray(index);

        ucf101_train_Features = randomSampleFeaturesPerClass;
        ucf101_train_FeaturesLabels = randomSampleLabelsPerClass;

    else
        % Cluster all the training features into clusters
        % If you want to cluster all the features
        ucf101_train_Features = ucf101_train_FeaturesArray;
        ucf101_train_FeaturesLabels = ucf101_train_FeaturesClassLabelArray;
    end
    
    tic;
    
    [ucf101_centers,ucf101_membership] = vl_kmeans(ucf101_train_Features', numClusters, 'verbose',...
                                    'algorithm', 'elkan', 'initialization', 'plusplus',...
                                    'maxnumiterations',numIter, 'numrepetitions', numReps);
    toc;
    % save tmp centers, membership
    save (sprintf('ucf101-tmp-stip-codebook.mat'), 'ucf101_centers', 'ucf101_membership');
end   

%% find the membership for all the remaining STIPs
% Skip these codes if you have ucf101-stip-codbook.mat
if sampleInd == 1
    % batch size 5047274 / 11 = 458843
    batch_size = int32(length(ucf101_train_FeaturesArray) / 11);
    % batch_array = [1, 458844, 917687, 1376530, 1835373, 2294216, 2753059,...
      %               3211902, 3670745, 4129588, 4588431, 5047274];
    batch_array = 1:batch_size:length(ucf101_train_FeaturesArray); 
        
    % pre-allocate
    ucf101_final_membership = zeros(1,batch_array(end));
    
    for b = 1:length(batch_array)
        fprintf('Batch Index %d start\n', batch_array(b));
        tic;
        if b == 1
            trainToClustersDist = vl_alldist2(ucf101_train_FeaturesArray(1, :)', ...
                                              ucf101_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            ucf101_final_membership = sortedInd(:,1)';
       
        else
            trainToClustersDist = vl_alldist2(ucf101_train_FeaturesArray(...
                                                batch_array(b-1)+1:batch_array(b), :)', ...
                                                ucf101_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            ucf101_final_membership = [ucf101_final_membership(1:batch_array(b-1)) ...
                                       sortedInd(:,1)'];
        end
        fprintf('size of membership: (%d, %d)\n\n', size(ucf101_final_membership));
        toc;
   end
end

save(sprintf('ucf101-STIP-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                sampleInd,numClusters,numIter,numReps), ...
                'ucf101_centers', 'ucf101_final_membership', 'numClusters', ...
                '-v7.3');
disp('Saving final codebook Done.');
%end

%% Now find the histograms for each of the videos
% fast to run
ucf101_train_VidNum = zeros(length(ucf101_train_ClassLabels),1);
for i = 1:length(ucf101_train_ClassLabels)
   ucf101_train_VidNum(i) = length(ucf101_train_ClassLabels{i});
end
ucf101_train_VidNumCumSum = cumsum(ucf101_train_VidNum);
ucf101_train_totalSeq = ucf101_train_VidNumCumSum(end);
ucf101_train_finalRepresentation = zeros(ucf101_train_totalSeq, numClusters);
ucf101_train_Labels = zeros(ucf101_train_totalSeq,1);

for i = 1:length(ucf101_train_ClassLabels)
    if i == 1
        ucf101_train_finalRepresentation(1,:) =  vl_ikmeanshist(numClusters,ucf101_final_membership(...
                   1:length(ucf101_train_SeqTotalFeatCumSum{1})));
        ucf101_train_Labels(i) = ucf101_train_ClassLabels{i};
    else
        ucf101_train_finalRepresentation(i,:) =  vl_ikmeanshist(numClusters,ucf101_final_membership(...
            ucf101_train_overallTotalFeatCumSum(i-1)+1 : ucf101_train_overallTotalFeatCumSum(i)));
        ucf101_train_Labels(i) = ucf101_train_ClassLabels{i};
    end
end

% Normalize histogram
ucf101_train_finalRepresentation_nor = (ucf101_train_finalRepresentation'./repmat(sum(ucf101_train_finalRepresentation'),numClusters,1))';

disp('Successfully Building BoVW in ucf101 using training set!')


%% Now for the test data (get BoVW representations for test set)
% Preload features if already computed
if exist(sprintf('ucf101_test_STIPs.mat'), 'file')
    disp('Loading STIPs features for testing set in UCF101 ...');
    load(sprintf('ucf101_test_STIPs.mat'));
else
    % Load STIP HOG-HOF features corresponding to these videos
    %offset = 5;
    ucf101_test_globalSeqCount = 0; 

    ucf101_test_DirList = dir([basePath, '/stip/test/*.txt']);

    ucf101_test_STIPFeaturesArray = cell(length(ucf101_test_DirList),1);
    ucf101_test_ClassLabels = cell(length(ucf101_test_DirList),1);
    ucf101_test_SeqTotalFeatNum = cell(length(ucf101_test_DirList),1);
    ucf101_test_SeqTotalFeatCumSum = cell(length(ucf101_test_DirList),1);
    ucf101_test_overallTotalFeatNum = zeros(length(ucf101_test_DirList),1);

    for fn = 1:length(ucf101_test_DirList)
       filename = ucf101_test_DirList(fn).name;
       % Load all STIP HOG-HOF for these
       STIPFilename = [filename(1:end-4), '.txt'];
       disp(STIPFilename);
       [ucf101_test_STIPLocation, ucf101_test_STIPSigma2,...
           ucf101_test_STIPTau2, ucf101_test_STIPDescriptor] = ...
           readHOGHOF_ucf([basePath '/stip/test/' STIPFilename]);

       % Now keep only those features that are in the shot boundaries ignoring
       % the first and last offset frames of each shot
       % Create feature cell array
       ucf101_test_stipFeaturesArray = cell(length(ucf101_test_STIPDescriptor),1);       
       ucf101_test_seqFeatNum = zeros(length(ucf101_test_STIPDescriptor),1);
       
       for i = 1:length(ucf101_test_STIPDescriptor)
           ucf101_test_stipFeaturesArray{i} = ucf101_test_STIPDescriptor(1,:);
           ucf101_test_seqFeatNum(i) = size(ucf101_test_stipFeaturesArray{i},1);
           ucf101_test_globalSeqCount = ucf101_test_globalSeqCount + size(ucf101_test_stipFeaturesArray{i},1);
       end
       
       ucf101_test_STIPFeaturesArray{fn} = ucf101_test_STIPDescriptor;
       
       % Re-Label (check every time) 
       % UCF-style (v_BoxingPunchingBag_g01_c01.avi)
       % Combine BoxingPunchingBag and BoxingSpeedBag together
       action_in_filename = STIPFilename(3:end-12);
       if (strcmp(action_in_filename, 'BoxingPunchingBag') || ...
          strcmp(action_in_filename, 'BoxingSpeedBag'))
           ucf101_test_ClassLabels{fn} = find(contains(actions, ...
                                                action_in_filename(1:6)));
       else
           ucf101_test_ClassLabels{fn} = find(contains(actions, ...
                                                        action_in_filename));
       end
              
       ucf101_test_SeqTotalFeatNum{fn} = ucf101_test_seqFeatNum;
       ucf101_test_SeqTotalFeatCumSum{fn} = cumsum(ucf101_test_seqFeatNum);
       ucf101_test_overallTotalFeatNum(fn) = ucf101_test_SeqTotalFeatCumSum{fn}(end);
    end
    ucf101_test_overallTotalFeatCumSum = cumsum(ucf101_test_overallTotalFeatNum);

    % Now accumulate all the features and quantize
    % Find total number of features
    disp(['Total number of HOG-HOF features for test set = ', int2str(ucf101_test_globalSeqCount)]);
    ucf101_test_FeaturesArray = zeros(ucf101_test_globalSeqCount, 72+90);    % For HOG(72)+HOF(90)
    ucf101_test_FeaturesClassLabelArray = zeros(ucf101_test_globalSeqCount,1);
    
    for i = 1:length(ucf101_test_ClassLabels)
        [r,c]= size(ucf101_test_STIPFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   ucf101_test_FeaturesArray(1:ucf101_test_SeqTotalFeatCumSum{1}(1),:) = ...
                       ucf101_test_STIPFeaturesArray{1}(1,1:c);
                   ucf101_test_FeaturesClassLabelArray(1:ucf101_test_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(ucf101_test_ClassLabels{1},ucf101_test_SeqTotalFeatNum{1}(1),1);
               else
                   ucf101_test_FeaturesArray(ucf101_test_SeqTotalFeatCumSum{1}(j-1)+1:ucf101_test_SeqTotalFeatCumSum{1}(j),:) = ...
                       ucf101_test_STIPFeaturesArray{1}(j,1:c);
                   ucf101_test_FeaturesClassLabelArray(ucf101_test_SeqTotalFeatCumSum{1}(j-1)+1:ucf101_test_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(ucf101_test_ClassLabels{1},ucf101_test_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    ucf101_test_FeaturesArray(ucf101_test_overallTotalFeatCumSum(i-1)+1:...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(1),:) = ...
                        ucf101_test_STIPFeaturesArray{i}(1,1:c);
                    ucf101_test_FeaturesClassLabelArray(ucf101_test_overallTotalFeatCumSum(i-1)+1:...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(ucf101_test_ClassLabels{i},ucf101_test_SeqTotalFeatNum{i}(1),1);
                else
                    ucf101_test_FeaturesArray( ...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(j),:) = ...
                        ucf101_test_STIPFeaturesArray{i}(j,1:c);
                    ucf101_test_FeaturesClassLabelArray(...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(ucf101_test_ClassLabels{i},ucf101_test_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
    
    save('ucf101_test_STIPs.mat', ...
         'ucf101_test_FeaturesArray', ...
         'ucf101_test_FeaturesClassLabelArray',...
         'ucf101_test_ClassLabels',...
         'ucf101_test_SeqTotalFeatNum',...
         'ucf101_test_SeqTotalFeatCumSum',...
         'ucf101_test_overallTotalFeatNum',...
         'ucf101_test_overallTotalFeatCumSum', ...
         '-v7.3');
     
    disp('Saving test features Done.');

end


%% Find memberships for all these points
if exist(sprintf('ucf101-STIP-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('ucf101-STIP-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    % batch size (1658235-1) / 314 =5281

    batch_size_test = int32(length(ucf101_test_FeaturesArray) / 314);
    batch_array_test = 1:batch_size_test:length(ucf101_test_FeaturesArray); 

    % pre-allocate
    ucf101_test_membership = zeros(1,batch_array_test(end));

    for b = 1:length(batch_array_test)
        fprintf('Batch Index %d start\n', batch_array_test(b));
        tic;
        if b == 1
            testToClustersDist = vl_alldist2(ucf101_test_FeaturesArray(1, :)', ...
                                              ucf101_centers);
            [testToClustersDist, test_sortedInd] = sort(testToClustersDist,2);
            ucf101_test_membership = test_sortedInd(:,1)';

        else
            testToClustersDist = vl_alldist2(ucf101_test_FeaturesArray(...
                                                batch_array_test(b-1)+1:batch_array_test(b), :)', ...
                                                ucf101_centers);
            [testToClustersDist, test_sortedInd] = sort(testToClustersDist,2);
            ucf101_test_membership = [ucf101_test_membership(1:batch_array_test(b-1)) ...
                                       test_sortedInd(:,1)'];
        end
        fprintf('size of test membership: (%d, %d)\n\n', size(ucf101_test_membership));
        toc;
    end

    % Now find the histograms for each of the test videos
    % Now find the histograms for each of the videos
    ucf101_test_allVidNum = zeros(length(ucf101_test_ClassLabels),1);
    for i = 1:length(ucf101_test_ClassLabels)
       ucf101_test_allVidNum(i) = length(ucf101_test_ClassLabels{i});
    end
    ucf101_test_VidNumCumSum = cumsum(ucf101_test_allVidNum);
    ucf101_test_totalTestSeq = ucf101_test_VidNumCumSum(end);
    ucf101_test_finalRepresentation = zeros(ucf101_test_totalTestSeq, numClusters);
    ucf101_test_Labels = zeros(ucf101_test_totalTestSeq,1);

    for i = 1:length(ucf101_test_ClassLabels)
        if i == 1
            ucf101_test_finalRepresentation(1,:) =  vl_ikmeanshist(numClusters,ucf101_test_membership(...
                       1:length(ucf101_test_SeqTotalFeatCumSum{1})));
            ucf101_test_Labels(i) = ucf101_test_ClassLabels{i};
        else
            ucf101_test_finalRepresentation(i,:) =  vl_ikmeanshist(numClusters,ucf101_test_membership(...
                ucf101_test_overallTotalFeatCumSum(i-1)+1 : ucf101_test_overallTotalFeatCumSum(i)));
            ucf101_test_Labels(i) = ucf101_test_ClassLabels{i};
        end
    end

    % Normalize histogram
    ucf101_test_finalRepresentation_nor = (ucf101_test_finalRepresentation'./ ...
                                            repmat(sum(ucf101_test_finalRepresentation'),...
                                            numClusters,1))';
                                        
    % Save train and test features
    save(sprintf('ucf101-STIP-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
        'ucf101_train_finalRepresentation','ucf101_train_finalRepresentation_nor',...
        'ucf101_test_finalRepresentation', 'ucf101_test_finalRepresentation_nor',...
        'ucf101_train_Labels', 'ucf101_test_Labels', 'actions',...
        'ucf101_centers', 'ucf101_final_membership', 'ucf101_test_membership');
    
    disp('Saving all BoVWs done.');
end

%% Final Feature mat 
if exist(sprintf('ucf101-STIP-allFeatures-%d-numclust.mat', numClusters), 'file')
   load(sprintf('ucf101-STIP-allFeatures-%d-numclust.mat', numClusters));
   disp('Loading ucf101-STIP-allfeatures.mat file done');

else
    
    ucf101.train.features = ucf101_train_finalRepresentation;
    ucf101.test.features = ucf101_test_finalRepresentation;
    ucf101.train.normalised_features = ucf101_train_finalRepresentation_nor;
    ucf101.test.normalised_features = ucf101_test_finalRepresentation_nor;
    ucf101.train.lables = ucf101_train_Labels;
    ucf101.test.lables = ucf101_test_Labels;
    
    ucf101.bovw.numClusters = numClusters;
    ucf101.bovw.sampleInd = sampleInd;
    ucf101.bovw.numIter = numIter;
    ucf101.bovw.numReps = numReps; 
    
    ucf101.bovw.actions = actions;
    
    save(sprintf('ucf101-STIP-allFeatures-%d-numclust.mat', numClusters),...
                    'ucf101');
    disp('Save all features, labels and parameters for train and test in ucf101')
end
disp('Everything is done !')
