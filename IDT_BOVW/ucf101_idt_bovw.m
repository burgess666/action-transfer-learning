% This script extracts all the IDT features for the training
% videos and quantizes these features to
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
addpath('IDT_BOVW');
addpath('utils');
% Set basic paths:
basePath= '/Volumes/Kellan/datasets/experimentTL/ucf101';
matPath = '/Volumes/Kellan/MATLAB/ActionRecogTL/';
addpath(matPath);


% Load all IDT features corresponding to these videos
% offset = 5;
ucf101_train_globalSeqCount = 0; 

% Preload features if already computed
if exist(sprintf([matPath 'ucf101_train_IDTs.mat']), 'file')
    load(sprintf([matPath 'ucf101_train_IDTs.mat']));
    disp('Loading IDTs features for training set in ucf101 done.');

elseif exist(sprintf([matPath 'ucf101_train_IDTs_read.mat']), 'file')
    load(sprintf([matPath 'ucf101_train_IDTs_read.mat']));
    disp('Loading idt_read.mat done.');
    
else
    ucf101_train_DirList = dir([basePath, '/idt/train/*.txt']);
    ucf101_train_IDTFeaturesArray = cell(length(ucf101_train_DirList),1);
    ucf101_train_ClassLabels = cell(length(ucf101_train_DirList),1);
    ucf101_train_SeqTotalFeatNum = cell(length(ucf101_train_DirList),1);
    ucf101_train_SeqTotalFeatCumSum = cell(length(ucf101_train_DirList),1);
    ucf101_train_overallTotalFeatNum = zeros(length(ucf101_train_DirList),1);

    for fn = 1:length(ucf101_train_DirList)
       filename = ucf101_train_DirList(fn).name;
       % Load all IDT HOG-HOF
       IDTFilename = [filename(1:end-4), '.txt'];
       disp(IDTFilename);
       ucf101_train_IDTDescriptor = readIDT([basePath,'/idt/train/',IDTFilename]);

       % Create feature cell array
       ucf101_train_idtFeaturesArray = cell(length(ucf101_train_IDTDescriptor),1);
       %classLabels = zeros(length(IDTDescriptor),1);
       ucf101_train_seqFeatNum = zeros(length(ucf101_train_IDTDescriptor),1);
       for i = 1:length(ucf101_train_IDTDescriptor)
           ucf101_train_idtFeaturesArray{i} = ucf101_train_IDTDescriptor(1,:);
           ucf101_train_seqFeatNum(i) = size(ucf101_train_idtFeaturesArray{i},1);
           ucf101_train_globalSeqCount = ucf101_train_globalSeqCount + size(ucf101_train_idtFeaturesArray{i},1);
       end
       ucf101_train_IDTFeaturesArray{fn} = ucf101_train_IDTDescriptor; 
       
       % Re-Label (check every time) 
       % UCF-style (v_BoxingPunchingBag_g01_c01.avi)
       % Combine BoxingPunchingBag and BoxingSpeedBag together
       action_in_filename = IDTFilename(3:end-12);
       if (strcmp(action_in_filename, 'BoxingPunchingBag') || ...
          strcmp(action_in_filename, 'BoxingSpeedBag'))
           ucf101_train_ClassLabels{fn} = find(contains(actions, ...
                                                action_in_filename(1:6)));
       else
           ucf101_train_ClassLabels{fn} = find(contains(actions, ...
                                                        action_in_filename));
       end
       
       ucf101_train_SeqTotalFeatNum{fn} = ucf101_train_seqFeatNum;
       ucf101_train_SeqTotalFeatCumSum{fn} = cumsum(ucf101_train_seqFeatNum);
       ucf101_train_overallTotalFeatNum(fn) = ucf101_train_SeqTotalFeatCumSum{fn}(end);
    end
    ucf101_train_overallTotalFeatCumSum = cumsum(ucf101_train_overallTotalFeatNum);
    % save
    save([matPath 'ucf101_train_IDTs_read.mat'], ...
         'ucf101_train_ClassLabels',...
         'ucf101_train_IDTFeaturesArray',...
         'ucf101_train_SeqTotalFeatNum',...
         'ucf101_train_SeqTotalFeatCumSum',...
         'ucf101_train_overallTotalFeatNum',...
         'ucf101_train_overallTotalFeatCumSum', ...
         '-v7.3');
end

   %% Now accumulate all the features and quantize
    % Starting First feature array from 1 to 3121025
    % Find total number of features
    ucf101_train_globalSeqCount = 7388073;
    disp(['Total number of IDT features = ', int2str(ucf101_train_globalSeqCount)]);
    % For trajectory(30) + HOG(96) + HOF(108) + MBH(96+96)
    % 7388073 * 426 is larger than memory
    % Divide train feature array into 2 sub-arrays
    
    sub_count1 = 0;
    for i=1:length(ucf101_train_ClassLabels)/2
        len = length(ucf101_train_SeqTotalFeatCumSum{i});
        sub_count1 = sub_count1 + len;
    end
    
    ucf101_train_FeaturesArray1 = zeros(sub_count1, 30+96+108+96+96);    
    ucf101_train_FeaturesClassLabelArray1 = zeros(sub_count1,1);
    
    disp('Starting First feature array from 1 to 3121025');
    % First feature array from 1 to 3121025
    for i = 1:length(ucf101_train_ClassLabels)/2
        disp(['The i is: ', int2str(i)]);
        [r,c]= size(ucf101_train_IDTFeaturesArray{i});
        if i == 1
           for j = 1:r
               disp(['The j is: ', int2str(j)]);
               if j == 1
                   ucf101_train_FeaturesArray1(1:ucf101_train_SeqTotalFeatCumSum{1}(1),:) = ...
                       ucf101_train_IDTFeaturesArray{1}(1,1:c);
                   ucf101_train_FeaturesClassLabelArray1(1:ucf101_train_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(ucf101_train_ClassLabels{1},ucf101_train_SeqTotalFeatNum{1}(1),1);
               else
                   ucf101_train_FeaturesArray1(ucf101_train_SeqTotalFeatCumSum{1}(j-1)+1:ucf101_train_SeqTotalFeatCumSum{1}(j),:) = ...
                       ucf101_train_IDTFeaturesArray{1}(j,1:c);
                   ucf101_train_FeaturesClassLabelArray1(ucf101_train_SeqTotalFeatCumSum{1}(j-1)+1:ucf101_train_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(ucf101_train_ClassLabels{1},ucf101_train_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                disp(['The j is: ', int2str(j)]);
                if j == 1
                    ucf101_train_FeaturesArray1(ucf101_train_overallTotalFeatCumSum(i-1)+1:...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(1),:) = ...
                        ucf101_train_IDTFeaturesArray{i}(1,1:c);
                    ucf101_train_FeaturesClassLabelArray1(ucf101_train_overallTotalFeatCumSum(i-1)+1:...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(ucf101_train_ClassLabels{i},ucf101_train_SeqTotalFeatNum{i}(1),1);
                else
                    ucf101_train_FeaturesArray1( ...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j),:) = ...
                        ucf101_train_IDTFeaturesArray{i}(j,1:c);
                    ucf101_train_FeaturesClassLabelArray1(...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j-1)+1:...
                        ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(ucf101_train_ClassLabels{i},ucf101_train_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
    disp('Completed First feature array from 1 to 3121025');
    %save
    save([matPath 'ucf101_train_FeaturesArray1.mat'], ...
                     'ucf101_train_FeaturesArray1', ...
                     'ucf101_train_FeaturesClassLabelArray1', ...
                     'ucf101_train_globalSeqCount', ...
                     '-v7.3');

    %% Starting Second array from 3121026 to 7388073
    disp('Starting Second array from 3121026 to 7388073');
    sub_count1 = 0;
    for i=1:length(ucf101_train_ClassLabels)/2
        len = length(ucf101_train_SeqTotalFeatCumSum{i});
        sub_count1 = sub_count1 + len;
    end
   
    ucf101_train_globalSeqCount = 7388073;
    sub_count2 = ucf101_train_globalSeqCount - sub_count1 -1 ;
    
    ucf101_train_FeaturesArray2 = zeros(sub_count2, 30+96+108+96+96);    
    ucf101_train_FeaturesClassLabelArray2 = zeros(sub_count2,1);
    
    previous_count = 3121025;
    
    % Second array from 3121026 to 7388073
    for i = (length(ucf101_train_ClassLabels)/2)+1 : length(ucf101_train_ClassLabels)
        [r,c]= size(ucf101_train_IDTFeaturesArray{i});
        disp(['The i is: ', int2str(i)]);
        for j = 1:r
            disp(['The j is: ', int2str(j)]);
            if j == 1
                ucf101_train_FeaturesArray2(ucf101_train_overallTotalFeatCumSum(i-1)+1-previous_count:...
                    ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(1)-previous_count,:) = ...
                    ucf101_train_IDTFeaturesArray{i}(1,1:c);
                ucf101_train_FeaturesClassLabelArray2(ucf101_train_overallTotalFeatCumSum(i-1)+1-previous_count:...
                    ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(1)-previous_count) = ...
                    repmat(ucf101_train_ClassLabels{i},ucf101_train_SeqTotalFeatNum{i}(1),1);
            else
                ucf101_train_FeaturesArray2( ...
                    ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j-1)+1-previous_count:...
                    ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j)-previous_count,:) = ...
                    ucf101_train_IDTFeaturesArray{i}(j,1:c);
                ucf101_train_FeaturesClassLabelArray2(...
                    ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j-1)+1-previous_count:...
                    ucf101_train_overallTotalFeatCumSum(i-1)+ucf101_train_SeqTotalFeatCumSum{i}(j)-previous_count) = ...
                    repmat(ucf101_train_ClassLabels{i},ucf101_train_SeqTotalFeatNum{i}(j),1);
            end
        end
    end
    disp('Completed Second array from 3121026 to 7388073');
    
    save([matPath 'ucf101_train_FeaturesArray2.mat'], ...
                 'ucf101_train_FeaturesArray2', ...
                 'ucf101_train_FeaturesClassLabelArray2', ...
                 '-v7.3');

    % save    
    save([matPath 'ucf101_train_IDTs.mat'], ...
         'ucf101_train_FeaturesArray1', ...
         'ucf101_train_FeaturesArray2', ...
         'ucf101_train_FeaturesClassLabelArray1',...
         'ucf101_train_FeaturesClassLabelArray2',...
         'ucf101_train_globalSeqCount', ...
         'ucf101_train_ClassLabels',...
         'ucf101_train_SeqTotalFeatNum',...
         'ucf101_train_SeqTotalFeatCumSum',...
         'ucf101_train_overallTotalFeatNum',...
         'ucf101_train_overallTotalFeatCumSum', ...
         '-v7.3');

    disp('Saving train features Done.');
%end

%% Loading ucf101_train_IDTs.mat
load(sprintf([matPath 'ucf101_train_IDTs.mat']));
disp('Loading done.')
%% Generate codebook
numClusters = 4000;
numIter = 8;
numReps = 1;
sampleInd = 1;  % Whether to sample data points or use all datapoints

% If clusters already precomputed, just load
if exist(sprintf('ucf101-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('ucf101-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
elseif exist(sprintf('ucf101-tmp-idt-codebook.mat'), 'file')
        load(sprintf('ucf101-tmp-idt-codebook.mat'));

else
    disp('Clustering (maybe long time to be consumed) ...');
    ucf101_train_FeaturesClassLabelArray1(ucf101_train_FeaturesClassLabelArray1==0)=1;
    ucf101_train_FeaturesClassLabelArray2(ucf101_train_FeaturesClassLabelArray2==0)=1;

    if sampleInd == 1
        % Multiplicative lagged Fibonacci generator
        s = RandStream('mlfg6331_64');
        numFeaturePointsPerClass = zeros(length(actions),1);
        
        for cat = 1:length(actions)
            numFeaturePointsPerClass(cat) = length(ucf101_train_FeaturesArray1 ...
                (ucf101_train_FeaturesClassLabelArray1==cat));
        end
                
        % Randomly sample 10% train features
        [randomSampleFeaturesPerClass1, index1] = datasample(s, ucf101_train_FeaturesArray1, ...
                                int32(length(ucf101_train_FeaturesArray1)*0.1),...
                                'Replace',false);
        randomSampleLabelsPerClass1 = ucf101_train_FeaturesClassLabelArray1(index1);
        
        [randomSampleFeaturesPerClass2, index2] = datasample(s, ucf101_train_FeaturesArray2, ...
                                int32(length(ucf101_train_FeaturesArray2)*0.1),...
                                'Replace',false);
        randomSampleLabelsPerClass2 = ucf101_train_FeaturesClassLabelArray2(index2);

        ucf101_train_Features = [randomSampleFeaturesPerClass1; randomSampleFeaturesPerClass2];
        ucf101_train_FeaturesLabels = [randomSampleLabelsPerClass1; randomSampleLabelsPerClass2];

        
    else
        % Cluster all the training features into clusters
        % If you want to cluster all the features
        %ucf101_train_Features = ucf101_train_FeaturesArray1;
        %ucf101_train_FeaturesLabels = ucf101_train_FeaturesClassLabelArray1;
        disp('Do not use all feature points for IDT, due to too large data.');
    end
    
    tic;

    [ucf101_centers,ucf101_membership] = vl_kmeans(ucf101_train_Features', numClusters, 'verbose',...
                                    'algorithm', 'elkan', 'initialization', 'plusplus',...
                                    'maxnumiterations',numIter, 'numrepetitions', numReps);
    toc;
    % save tmp centers, membership
    save (sprintf('ucf101-tmp-idt-codebook.mat'), 'ucf101_centers', 'ucf101_membership');
end

%% find the membership for all the remaining IDTs
% Skip these codes if you have ucf101-idt-codbook.mat
% feature array 1 has 3121025, feature array 2 has 4267048

if sampleInd == 1
    % Must check the number of feature array !!
    % batch size (3121025-1) / 4736 = 659 
    batch_size1 = int32(length(ucf101_train_FeaturesArray1) / 4736);
    batch_array1 = 1:batch_size1:length(ucf101_train_FeaturesArray1);
    
    % batch size (4267048-1) / 3669  = 1163
    batch_size2 = int32(length(ucf101_train_FeaturesArray2) / 3669);
    batch_array2 = 1:batch_size2:length(ucf101_train_FeaturesArray2);
        
    % pre-allocate
    ucf101_final_membership1 = zeros(1,batch_array1(end));
    ucf101_final_membership2 = zeros(1,batch_array2(end));

    
    for b = 1:length(batch_array1)
        fprintf('Batch Index %d start\n', batch_array1(b));
        tic;
        if b == 1
            trainToClustersDist = vl_alldist2(ucf101_train_FeaturesArray1(1, :)', ...
                                              ucf101_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            ucf101_final_membership1 = sortedInd(:,1)';
       
        else
            trainToClustersDist = vl_alldist2(ucf101_train_FeaturesArray1(...
                                                batch_array1(b-1)+1:batch_array1(b), :)', ...
                                                ucf101_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            ucf101_final_membership1 = [ucf101_final_membership1(1:batch_array1(b-1)) ...
                                       sortedInd(:,1)'];
        end
        fprintf('size of membership: (%d, %d)\n\n', size(ucf101_final_membership1));
        toc;
        disp('Completed first array.');
    end
   
    for b = 1:length(batch_array2)
        fprintf('Batch Index %d start\n', batch_array2(b));
        tic;
        if b == 1
            trainToClustersDist = vl_alldist2(ucf101_train_FeaturesArray2(1, :)', ...
                                              ucf101_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            ucf101_final_membership2 = sortedInd(:,1)';
       
        else
            trainToClustersDist = vl_alldist2(ucf101_train_FeaturesArray2(...
                                                batch_array2(b-1)+1:batch_array2(b), :)', ...
                                                ucf101_centers);
            [trainToClustersDist, sortedInd] = sort(trainToClustersDist,2);
            ucf101_final_membership2 = [ucf101_final_membership2(1:batch_array2(b-1)) ...
                                       sortedInd(:,1)'];
        end
        fprintf('size of membership: (%d, %d)\n\n', size(ucf101_final_membership2));
        toc;
        disp('Completed second array.');
    end
    ucf101_final_membership = [ucf101_final_membership1, ucf101_final_membership2];
end

save(sprintf('ucf101-IDT-codebook-clustered-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                sampleInd,numClusters,numIter,numReps), ...
                'ucf101_centers', 'ucf101_final_membership', 'numClusters', ...
                '-v7.3');
disp('Saving final codebook done.');
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
ucf101_train_finalRepresentation_nor = (ucf101_train_finalRepresentation'./ ...
                                        repmat(sum(ucf101_train_finalRepresentation'),numClusters,1))';

disp('Successfully Building BoVW in ucf101 using training set!')


%% Now for the test data (get BoVW representations for test set)
% Preload features if already computed
if exist(sprintf([matPath 'ucf101_test_IDTs.mat']), 'file')
    load(sprintf([matPath 'ucf101_test_IDTs.mat']));
    disp('Loading IDTs features for testing set in ucf101 done.');

else
    % Load IDT features corresponding to these videos
    %offset = 5;
    ucf101_test_globalSeqCount = 0; 

    ucf101_test_DirList = dir([basePath,'/idt/test/*.txt']);

    ucf101_test_IDTFeaturesArray = cell(length(ucf101_test_DirList),1);
    ucf101_test_ClassLabels = cell(length(ucf101_test_DirList),1);
    ucf101_test_SeqTotalFeatNum = cell(length(ucf101_test_DirList),1);
    ucf101_test_SeqTotalFeatCumSum = cell(length(ucf101_test_DirList),1);
    ucf101_test_overallTotalFeatNum = zeros(length(ucf101_test_DirList),1);

    for fn = 1:length(ucf101_test_DirList)
       filename = ucf101_test_DirList(fn).name;
       % Load all IDT for these
       IDTFilename = [filename(1:end-4), '.txt'];
       disp(IDTFilename);
       ucf101_test_IDTDescriptor = readIDT([basePath,'/idt/test/',IDTFilename]);

       % Now keep only those features that are in the shot boundaries ignoring
       % the first and last offset frames of each shot
       % Create feature cell array
       ucf101_test_idtFeaturesArray = cell(length(ucf101_test_IDTDescriptor),1);       
       ucf101_test_seqFeatNum = zeros(length(ucf101_test_IDTDescriptor),1);
       
       for i = 1:length(ucf101_test_IDTDescriptor)
           ucf101_test_idtFeaturesArray{i} = ucf101_test_IDTDescriptor(1,:);
           ucf101_test_seqFeatNum(i) = size(ucf101_test_idtFeaturesArray{i},1);
           ucf101_test_globalSeqCount = ucf101_test_globalSeqCount + size(ucf101_test_idtFeaturesArray{i},1);
       end
       
       ucf101_test_IDTFeaturesArray{fn} = ucf101_test_IDTDescriptor;
       % Re-Label (check every time) 
       % UCF-style (v_BoxingPunchingBag_g01_c01.avi)
       % Combine BoxingPunchingBag and BoxingSpeedBag together
       action_in_filename = IDTFilename(3:end-12);
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
    disp(['Total number of IDT features for test set = ', int2str(ucf101_test_globalSeqCount)]);
    % For trajectory(30) + HOG(96) + HOF(108) + MBH(96+96)
    ucf101_test_FeaturesArray = zeros(ucf101_test_globalSeqCount, 30+96+108+96+96);
    ucf101_test_FeaturesClassLabelArray = zeros(ucf101_test_globalSeqCount,1);
    
    for i = 1:length(ucf101_test_ClassLabels)
        [r,c]= size(ucf101_test_IDTFeaturesArray{i});
        if i == 1
           for j = 1:r
               if j == 1
                   ucf101_test_FeaturesArray(1:ucf101_test_SeqTotalFeatCumSum{1}(1),:) = ...
                       ucf101_test_IDTFeaturesArray{1}(1,1:c);
                   ucf101_test_FeaturesClassLabelArray(1:ucf101_test_SeqTotalFeatCumSum{1}(1)) = ...
                       repmat(ucf101_test_ClassLabels{1},ucf101_test_SeqTotalFeatNum{1}(1),1);
               else
                   ucf101_test_FeaturesArray(ucf101_test_SeqTotalFeatCumSum{1}(j-1)+1:ucf101_test_SeqTotalFeatCumSum{1}(j),:) = ...
                       ucf101_test_IDTFeaturesArray{1}(j,1:c);
                   ucf101_test_FeaturesClassLabelArray(ucf101_test_SeqTotalFeatCumSum{1}(j-1)+1:ucf101_test_SeqTotalFeatCumSum{1}(j)) = ...
                       repmat(ucf101_test_ClassLabels{1},ucf101_test_SeqTotalFeatNum{1}(j),1);
               end
           end
        else
            for j = 1:r
                if j == 1
                    ucf101_test_FeaturesArray(ucf101_test_overallTotalFeatCumSum(i-1)+1:...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(1),:) = ...
                        ucf101_test_IDTFeaturesArray{i}(1,1:c);
                    ucf101_test_FeaturesClassLabelArray(ucf101_test_overallTotalFeatCumSum(i-1)+1:...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(1)) = ...
                        repmat(ucf101_test_ClassLabels{i},ucf101_test_SeqTotalFeatNum{i}(1),1);
                else
                    ucf101_test_FeaturesArray( ...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(j),:) = ...
                        ucf101_test_IDTFeaturesArray{i}(j,1:c);
                    ucf101_test_FeaturesClassLabelArray(...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(j-1)+1:...
                        ucf101_test_overallTotalFeatCumSum(i-1)+ucf101_test_SeqTotalFeatCumSum{i}(j)) = ...
                        repmat(ucf101_test_ClassLabels{i},ucf101_test_SeqTotalFeatNum{i}(j),1);
                end
            end
        end
    end
    
    save([matPath 'ucf101_test_IDTs.mat'], ...
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
if exist(sprintf('ucf101-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps), 'file')
    load(sprintf('ucf101-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
        sampleInd,numClusters,numIter,numReps));
else
    % check total number of test feature array    
    % batch size (3234021-1) / 8005 =404

    batch_size_test = int32((length(ucf101_test_FeaturesArray)-1) / 8005);
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
    save(sprintf('ucf101-IDT-allBoVWs-sampled-%d-numclust-%d-numIter-%d-numReps-%d.mat',...
                    sampleInd,numClusters,numIter,numReps), ...
        'ucf101_train_finalRepresentation','ucf101_train_finalRepresentation_nor',...
        'ucf101_test_finalRepresentation', 'ucf101_test_finalRepresentation_nor',...
        'ucf101_train_Labels', 'ucf101_test_Labels', 'actions');
    
    disp('Saving all BoVWs Done.');

end

%% Final Feature mat 
if exist(sprintf('ucf101-IDT-allFeatures-%d-numclust.mat',numClusters), 'file')
    load(sprintf('ucf101-IDT-allFeatures-%d-numclust.mat', numClusters));
    disp('Loading ucf101-IDT-allfeatures.mat file done');
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
    
    save(sprintf('ucf101-IDT-allFeatures-%d-numclust.mat', numClusters),...
                    'ucf101');  
    disp('Save all features, labels and parameters for train and test in ucf101')
end
disp('Everything is done !')
