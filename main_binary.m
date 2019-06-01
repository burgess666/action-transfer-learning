% This is a demo file that shows how to use A-SVM, PMT-SVM, DA-SVM
% on a simple classification example. 

% MOSEK optimization toolkit is advised for faster QP optimization which is
% used in Model Transfer SVM learning procedures.
for r=1:5
    repeat_str = int2str(r);
    addpath svms
    addpath utils
    addpath STIP_BOVW

    numClusters = 4000;

    % Be careful to change it when using different datasets
    % weizmann, kth, hmdb51, ucf101

    % Change line 20, 21, 43, 44, 218

    source_string = 'weizmann';
    target_string = 'ucf101';

    % 'IDT' or 'STIP'
    feature = 'STIP'; 
    %feature = 'IDT'; 

    if (exist(sprintf([source_string '-' feature '-allFeatures-%d-numclust.mat'], numClusters)))
        load(sprintf([source_string '-' feature '-allFeatures-%d-numclust.mat'], numClusters));
        disp(['Loading ' source_string ' done.']);
    else
        disp(['Must run ' source_string '_' lower(feature) '_bovw.m at first!'])
        disp(['Running ' source_string '_' lower(feature) '_bovw.m...'])
    end  

    if (exist(sprintf([target_string '-' feature '-allFeatures-%d-numclust.mat'], numClusters)))
        load (sprintf([target_string '-' feature '-allFeatures-%d-numclust.mat'], numClusters));
        disp(['Loading ' target_string ' done.']);
    else
        disp(['Must run ' target_string '_' lower(feature) '_stip_bovw.m at first!'])
        disp(['Running ' target_string '_' lower(feature) '_stip_bovw.m...'])
    end

    % Be careful to change it when using different datasets
    source = weizmann;
    target = ucf101;
    % Resampling

    % Change the x_actions when using different datasets
    common_category = intersect(source.bovw.actions, ...
                                target.bovw.actions);

    source_train_count = 0;
    source_test_count = 0;
    target_train_count = 0;
    target_test_count = 0;

    % Count length for train and test in source and target
    for i=1:length(common_category)
        index_source_actionName = find(strcmp(source.bovw.actions, common_category(i)));
        index_source_train_origin = find(source.train.lables==index_source_actionName);
        len = length(index_source_train_origin);
        source_train_count = source_train_count + len;

        index_source_test_origin = find(source.test.lables==index_source_actionName);
        len = length(index_source_test_origin);
        source_test_count = source_test_count + len;


        index_target_actionName = find(strcmp(target.bovw.actions, common_category(i)));
        index_target_train_origin = find(target.train.lables==index_target_actionName);
        len = length(index_target_train_origin);
        target_train_count = target_train_count + len;

        index_target_test_origin = find(target.test.lables==index_target_actionName);
        len = length(index_target_test_origin);
        target_test_count = target_test_count + len;

    end

    % Init ReSample
    source.ReSample.train.features = [];
    source.ReSample.train.features = [];
    source.ReSample.train.nor_features = [];
    source.ReSample.train.labels = [];

    source.ReSample.test.features = [];
    source.ReSample.test.nor_features = [];
    source.ReSample.test.labels = [];

    target.ReSample.train.features = [];
    target.ReSample.train.nor_features = [];
    target.ReSample.train.labels = [];

    target.ReSample.test.features = [];
    target.ReSample.test.nor_features = [];
    target.ReSample.test.labels = [];

    % Re-Sampling source and target in train and test
    for i=1:length(common_category)
        % source train
        index_source_actionName = find(strcmp(source.bovw.actions, common_category(i)));
        index_source_train_origin = find(source.train.lables==index_source_actionName);
        source.ReSample.train.features = [source.ReSample.train.features ; ...
                                        source.train.features(index_source_train_origin,:)];
        source.ReSample.train.nor_features = [source.ReSample.train.nor_features;...
                                                source.train.normalised_features(index_source_train_origin,:)];
        % for labels
        if i==1
            source.ReSample.train.labels(1:length(index_source_train_origin),1) = i;
        else
            len = length(source.ReSample.train.labels);
            source.ReSample.train.labels(len+1:length(index_source_train_origin)+len,1) = i;
        end


        % source test
        index_source_test_origin = find(source.test.lables==index_source_actionName);
        source.ReSample.test.features = [source.ReSample.test.features;...
                                        source.test.features(index_source_test_origin,:)];
        source.ReSample.test.nor_features = [source.ReSample.test.nor_features;...
                                                source.test.normalised_features(index_source_test_origin,:)];

        % for labels
        if i==1
            source.ReSample.test.labels(1:length(index_source_test_origin),1) = i;
        else
            len = length(source.ReSample.test.labels);
            source.ReSample.test.labels(len+1:length(index_source_test_origin)+len,1) = i;
        end

        % target train
        index_target_actionName = find(strcmp(target.bovw.actions, common_category(i)));
        index_target_train_origin = find(target.train.lables==index_target_actionName);
        target.ReSample.train.features = [target.ReSample.train.features;...
                                            target.train.features(index_target_train_origin,:)];
        target.ReSample.train.nor_features = [target.ReSample.train.nor_features;...
                                               target.train.normalised_features(index_target_train_origin,:)];
        % for labels
        if i==1
            target.ReSample.train.labels(1:length(index_target_train_origin),1) = i;
        else
            len = length(target.ReSample.train.labels);
            target.ReSample.train.labels(len+1:length(index_target_train_origin)+len,1) = i;
        end

        % target test
        index_target_test_origin = find(target.test.lables==index_target_actionName);
        target.ReSample.test.features = [target.ReSample.test.features ; target.test.features(index_target_test_origin,:)];
        target.ReSample.test.nor_features = [target.ReSample.test.nor_features ; target.test.normalised_features(index_target_test_origin,:)];
        % for labels
        if i==1
            target.ReSample.test.labels(1:length(index_target_test_origin),1) = i;
        else
            len = length(target.ReSample.test.labels);
            target.ReSample.test.labels(len+1:length(index_target_test_origin)+len,1) = i;
        end
    end
    % End: Re-sampling and Re-label

    % Start: Train on source, test on source %
   
    ws_zero = zeros(numClusters,1);
    C=0.002;
    source.ReSample.train.labels(source.ReSample.train.labels==1,1) = 1;
    source.ReSample.train.labels(source.ReSample.train.labels~=1,1) = -1;

    target.ReSample.train.labels(target.ReSample.train.labels==1,1) = 1;
    target.ReSample.train.labels(target.ReSample.train.labels~=1,1) = -1;

    source.svm_ss = A_SVM(source.ReSample.train.labels,...
                                source.ReSample.train.features,...
                                C, ws_zero);
    scores_OnSource = source.ReSample.test.features * source.svm_ss.w ...
                                        + source.svm_ss.b;

    % l2 normalization
    source.svm_ss.w = source.svm_ss.w / norm(source.svm_ss.w(:));

    predict_ss = sign(scores_OnSource);
    predict_ss(predict_ss==0) = 1;
    predict_ss(predict_ss==-1) = 2;
    stat_ss = confusionmatStats(source.ReSample.test.labels, predict_ss);


    % End: Train on source, test on source %

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Start: Train on source, and directly test on target %

    % Directly evaluate on target test part. 
    % (use the same model as above)
    scores_OnTarget = target.ReSample.test.features * source.svm_ss.w ...
                                        + source.svm_ss.b;
    predict_st = sign(scores_OnTarget);
    predict_st(predict_st==0) = 1;
    predict_st(predict_st==-1) = 2;
    stat_st = confusionmatStats(target.ReSample.test.labels, predict_st);

    % End: Train on source, and directly test on target %

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Start: Train & test target classifier with increasing number of samples

    % Increasing number of examples (5)
    stepSize = 0.05;
    idx = [];
    count_perCat = [];

    count_perCat{1} = length(find(target.ReSample.train.labels==1));
    index{1} = find(target.ReSample.train.labels==1);

    count_perCat{2} = length(find(target.ReSample.train.labels==-1));
    index{2} = find(target.ReSample.train.labels==-1);


    min_cat = min(cell2mat(count_perCat));

    for i=1:length(common_category)
        step_count = 0;
        for step=round(min_cat * [stepSize:stepSize:1])
            % step can start from ceil(count_perCat*stepSize)
            index{i} = index{i}(randperm(count_perCat{i}));
            step_count = step_count + 1;
            idx{i}(1:step,step_count) = index{i}(1:step);
        end
    end

    step_count = 0;
    % start from 5 samples
    for step=round(min_cat * [stepSize:stepSize:1])
        % step can start from ceil(count_perCat*stepSize)
        fprintf('\t %d sample(s)\n',step);
        pause(0.01);
        step_count = step_count + 1;
        % change when having more or less common category
        target_index = [idx{1}(1:step,step_count);
                        idx{2}(1:step,step_count)
                        ];

        % Non-transfer (re-train target training data along with source training data, and test on target testing data )
        target.non_svm{step_count} = A_SVM([source.ReSample.train.labels; target.ReSample.train.labels(target_index)],...
                                            [source.ReSample.train.features; target.ReSample.train.features(target_index,:)],...)
                                            C, ws_zero);

        scores_non = target.ReSample.test.features * target.non_svm{step_count}.w ...
                                        + target.non_svm{step_count}.b;
        predict_nonsvm = sign(scores_non);
        predict_nonsvm(predict_nonsvm==0) = 1;
        predict_nonsvm(predict_nonsvm==-1) = 2;
        stat_non_svm(step_count) = confusionmatStats(target.ReSample.test.labels, predict_nonsvm);

        % A_SVM            
        target.a_svm(step_count) = A_SVM(target.ReSample.train.labels(target_index),...
                                        target.ReSample.train.features(target_index,:), ...
                                         C, source.svm_ss.w);

        scores_asvm = target.ReSample.test.features * target.a_svm(step_count).w ...
                                        + target.a_svm(step_count).b;
        predict_asvm = sign(scores_asvm);
        predict_asvm(predict_asvm==0) = 1;
        predict_asvm(predict_asvm==-1) = 2;

        stat_a_svm(step_count) = confusionmatStats(target.ReSample.test.labels, predict_asvm);

        % PMT_SVM            
        target.pmt_svm(step_count) = PMT_SVM(target.ReSample.train.labels(target_index),...
                                            target.ReSample.train.features(target_index,:), ...
                                            C, source.svm_ss.w);
        scores_pmt{step_count} = target.ReSample.test.features * target.pmt_svm(step_count).w ...
                                        + target.pmt_svm(step_count).b;
        predict_pmt = sign(scores_pmt{step_count});
        predict_pmt(predict_pmt==0) = 1;
        predict_pmt(predict_pmt==-1) = 2;
        stat_pmt_svm(step_count) = confusionmatStats(target.ReSample.test.labels, predict_pmt);

    end

    % End: Train & test target classifier with increasing number of samples %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Recall
    %source_string = 'kth';
    %target_string = 'hmdb51';
    %feature = 'IDT';
    metrics = 'recall';
    mean_recall_non_svm = zeros(step_count,1);
    mean_recall_a_svm = zeros(step_count,1);
    mean_recall_pmt_svm = zeros(step_count,1);
    mean_recall_ss = mean(stat_ss.recall);
    mean_recall_st = mean(stat_st.recall);

    for i=1:step_count
        mean_recall_non_svm(i) = mean(stat_non_svm(i).recall);
        mean_recall_a_svm(i) =  mean(stat_a_svm(i).recall);
        mean_recall_pmt_svm(i) =  mean(stat_pmt_svm(i).recall);
    end

    % Draw comparison figure between SVM, A-SVM, PMT-SVM
    fprintf('Drawing the Recall curves.\n');
    drawComparisonFigure( ['Source: ' source_string ' Target: ' target_string], ...
        round(min_cat * [stepSize:stepSize:1]),{ ...
        mean_recall_pmt_svm(:,1),['PMT-SVM']; ...
        mean_recall_a_svm(:,1),['A-SVM']; ...
        mean_recall_non_svm(:,1),['Non-transfer SVM']; ...
        ones(length(round(min_cat * [stepSize:stepSize:1],1)))*mean_recall_st,...
            ['Source SVM (Test on Target)']
        });

    fprintf('End Drawing.\n');

    % Save results
    save(sprintf([repeat_str '-3-Results-' source_string '-' target_string '-' feature '-' metrics '-stepsize-%.2f.mat'],stepSize), ...
                'stat_ss', 'stat_st', 'stat_a_svm', 'stat_non_svm', 'stat_pmt_svm',...
                'mean_recall_ss', 'mean_recall_st', 'mean_recall_a_svm','mean_recall_non_svm', 'mean_recall_pmt_svm',...
                'step_count', 'stepSize', 'min_cat', 'source_string', 'target_string');
    clear;
    clc;
end




           
        
%% OLD Experiments
%{
% Combine train and test in source (removed)
%source_combined = cell2struct(cellfun(@vertcat,struct2cell(source_ucf.train),...
 %                   struct2cell(source_ucf.test),'uni',0),...
  %                   fieldnames(source_ucf.train),1);
                 
% change size based on the size of codebook
ws_zero = zeros(numClusters,1);
C=0.002;
% Change the x_actions when using different datasets
common_category = intersect(target.bovw.actions, ...
                            source.bovw.actions);
% add tmp label for binary classification ( 1 and -1 )
source.train.tmplabels = ones(length(source.train.lables),1);

ap_OnSource_CV = zeros(length(common_category), 1);
APs_TargetSamplesOnly = [];
APs_ASVM = [];
APs_PMTSVM = [];

for cat = 1:length(common_category)
    
    % Start: Train on source, test on source %
    
    source.train.tmplabels(source.train.lables==cat,1) = 1;
    source.train.tmplabels(source.train.lables~=cat,1) = -1;
    source.test.tmplabels(source.test.lables==cat,1) = 1;
    source.test.tmplabels(source.test.lables~=cat,1) = -1;
    
    target.train.tmplabels(target.train.lables==cat,1) = 1;
    target.train.tmplabels(target.train.lables~=cat,1) = -1;
    target.test.tmplabels(target.test.lables==cat,1) = 1;
    target.test.tmplabels(target.test.lables~=cat,1) = -1;
    
    % Train on source and test on source
    source.svm(cat) = A_SVM(source.train.tmplabels,...
                                source.train.features,...
                                C, ws_zero);
    scores_OnSource = source.test.features * source.svm(cat).w ...
                                        + source.svm(cat).b;

    %acc_OnSource(ceil(i)) = calCM(source_combined.lables(source_combined.cvp_indices==i)', scores_OnSource');                                              
    ap_OnSource_CV(cat) = computeAP(scores_OnSource,source.test.tmplabels);
    % mAP
    mean_ap_OnSource = sum(ap_OnSource_CV)/cat;
    % End: Train on source, test on source %
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Start: Train on source, and directly test on target %
        
    % Directly evaluate on target test part. 
    % (use the same model as above)
    scores_onTarget = target.test.features * source.svm(cat).w ...
                        + source.svm(cat).b;
                    
    %acc_source = calCM(target_hdmb.test.lables', scores_onTarget');
    ap_OnTarget_NonTransfer(cat) = computeAP(scores_onTarget,...
                                    target.test.tmplabels);

    % l2 normalization
    source.svm(cat).w = source.svm(cat).w / ...
                            norm(source.svm(cat).w(:)); 
    
% End: Train on source, and directly test on target %

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start: Train & test target classifier with increasing number of samples %
    
    % Increasing number of examples
    stepSize = 5;
    
    pos_count = length(find(target.train.tmplabels==1));
    pos_index = find(target.train.tmplabels==1);
    neg_count = length(find(target.train.tmplabels==-1));
    neg_index = find(target.train.tmplabels==-1);

    % Train on Target, and test on Target    
    for step=1:stepSize:pos_count
    
        fprintf('\t %d sample(s)\n',step);
        pause(0.001);
        
        pos_index = pos_index(randperm(pos_count));
        neg_index = neg_index(randperm(neg_count));
        
        % ??? balanced positive and negative samples 
        idx = [pos_index(1:step);neg_index(1:step)];    
        % unbalanced positive and negative samples
        %idx = [1:neg_count+i];  

        % target SVM trained with using target samples only
        target.svm(cat) = A_SVM(target.train.tmplabels(idx),...
                             target.train.features(idx,:),...
                             C, ws_zero);
        scores_TargetSamplesOnly = target.test.features * target.svm(cat).w ...
                                                    + target.svm(cat).b;
                                         
        APs_TargetSamplesOnly(ceil(step/stepSize),cat) = computeAP(scores_TargetSamplesOnly, ...
                                                             target.test.tmplabels);    
        %ACCs(ceil(i/stepSize),1) = calCM(target_hdmb.test.lables', scores');


        % A-SVM
        target.a_svm(cat) = A_SVM(target.train.tmplabels(idx), ...
                               target.train.features(idx,:), ...
                               C, source.svm(cat).w);
                           
        scores_ASVM = target.test.features * target.a_svm(cat).w ...
                                        + target.a_svm(cat).b;
                                         
        APs_ASVM(ceil(step/stepSize),cat) = computeAP(scores_ASVM, ...
                                                target.test.tmplabels);
   
        %ACCs(ceil(i/stepSize),2) = calCM(target_hdmb.test.lables', scores');

        % PMT-SVM
        target.pmt_svm(cat) = PMT_SVM(target.train.tmplabels(idx),...
                                   target.train.features(idx,:),...
                                   C, source.svm(cat).w);
        scores_PMTSVM = target.test.features * target.pmt_svm(cat).w ...
                                          + target.pmt_svm(cat).b;
                                         
        APs_PMTSVM(ceil(step/stepSize),cat) = computeAP(scores_PMTSVM, ...
                                                    target.test.tmplabels);

        %ACCs(ceil(i/stepSize),3) = calCM(target_hdmb.test.lables', scores');
    end

% End: Train & test target classifier with increasing number of samples %
end

% Draw comparison figure between SVM, A-SVM, PMT-SVM
fprintf('Drawing the AP curves.\n');
drawComparisonFigure( ['Source: WeizMann' ' Target: KTH'] , ...
    [1:stepSize:pos_count],{ ...
    APs_PMTSVM(:,1),['PMT-SVM']; ...
    APs_ASVM(:,1),['A-SVM']; ...
    APs_TargetSamplesOnly(:,1),['Target SVM (Test on Target)']; ...
    ones(length(1:stepSize:pos_count),1)*ap_OnTarget_NonTransfer(2),['Source SVM (Test on Target)']; ...
    ones(length(1:stepSize:pos_count),1)*ap_OnSource_CV(2), ['Source SVM (Test on Source)']
    });
fprintf('End Drawing.\n');

%}