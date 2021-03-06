classdef svm
    methods (Static)
        function Model=train(features, labels, C, svmStruct, algorithm, varargin)
            %{
                Input: train feature
                       labels
                       C
                       weight (ws_zero or source.svm.w)
            
                Output:
                        Model
            %}
            classIndex=unique(labels);
            sumValue=sum(classIndex);
            nsample=length(classIndex);
            
            if nsample>2
                model=cell(1,nsample);
                for i=1:nsample
                    classx=labels;
                    classx(classx==classIndex(i))=sumValue;
                    classx(classx~=sumValue)=-1;
                    classx(classx==sumValue)=1;
                    
                    %len_pos = length(classx(classx==1));                 
                    %pos_index = find(classx==1);
                    %neg_index = find(classx==-1);
                    
                    %neg_bal_index = neg_index(randperm(len_pos));
                    
                    %classx_bal = [classx(pos_index);classx(neg_bal_index)];                    
                    %features_bal = [features(pos_index,:);features(neg_bal_index,:)];
                    
                    
                    if strcmp(algorithm,'A_SVM')
                        % svmStruct = ws_zero or source.svm.w
                        if ~isstruct(svmStruct)
                            model{i} = A_SVM(classx, features, C, svmStruct);
                        else
                            model{i} = A_SVM(classx, features, C, svmStruct.model{i}.w);
                        end
                    elseif strcmp(algorithm,'PMT_SVM')
                        if ~isstruct(svmStruct)
                            model{i} = PMT_SVM(classx, features, C, svmStruct);
                        else
                            model{i} = PMT_SVM(classx, features, C, svmStruct.model{i}.w);
                        end
                    else
                        error (message('stats:train:UnknownAlgorithm'))
                    end
                    fprintf('Multi Class SVM Model for Class Instance %d --->\n',classIndex(i))
                    disp(model{i})
                end
            else
                classx=labels;
                classx(classx~=1)=-1;
                classx(classx==1)=1;
                %len_pos = length(classx(classx==1));                 
                %pos_index = find(classx==1);
                %neg_index = find(classx==-1);

                %neg_bal_index = neg_index(randperm(len_pos));

                %classx_bal = [classx(pos_index);classx(neg_bal_index)];                    
                %features_bal = [features(pos_index);features(neg_bal_index)];

                if strcmp(algorithm,'A_SVM')
                    % svmStruct = ws_zero or source.svm.w
                    if ~isstruct(svmStruct)
                        model = A_SVM(classx, features, C, svmStruct);
                    else
                        model = A_SVM(classx, features, C, svmStruct.model.w);
                    end
                elseif strcmp(algorithm,'PMT_SVM')
                    if ~isstruct(svmStruct)
                        model = PMT_SVM(classx, features, C, svmStruct);
                    else
                        model = PMT_SVM(classx, features, C, svmStruct.model.w);
                    end
                else
                    error (message('stats:train:UnknownAlgorithm'))               
                end
               fprintf('\n Two class svm  Model--->\n')
               disp(model)
            end
            Model.model=model;
            Model.classInstance=classIndex;
            fprintf('\nTrain Model Completed !\n')
        
        end
        
        
        % Predict function
        function output=predict(Model,sample,varargin)
            model=Model.model;
            classIndex=Model.classInstance;
            nsample=length(classIndex);
            if nsample>2
                numberOfSamples=size(sample,1);
                classRange=zeros(numberOfSamples,length(classIndex));
                for i=1:nsample
                    [~,threshold]=svm.svmclassify(model{i},sample,varargin{:});
                    classRange(:,i)=threshold;
                    fprintf('\nMulti Class SVM classify values calculated for Class Instance %d ',classIndex(i));
                end
                [~,index]=max(transpose(classRange));
                output=classIndex(index);
            else
                output=svm.svmclassify(model,sample,varargin{:});
                output(output==-1)=2;
            end
            fprintf('\n SVM Classification is completed\n')
        end
        
        %{ 
                Function: svmclassify
                Input:  svm model
                        test_features
                Output: predicted class
                        scores
        %} 
        function [predicted_class,score] = svmclassify(svmStruct, sample, varargin)
            
            % deal with struct input case
            if ~isstruct(svmStruct)
                error(message('stats:svmclassify:TwoInputsNoStruct'));
            end
            
            if ~isnumeric(sample) || ~ismatrix(sample)
                error(message('stats:svmclassify:BadSample'));
            end
            
            labels = svmStruct.labels;
            
            % check group is a vector -- though char input is special...
            if ~isvector(labels) && ~ischar(labels)
                error(message('stats:svmclassify:GroupNotVector'));
            end

            % do the classification
            if ~isempty(sample)
               
                try
                    [predicted_class,score] = svm.svmdecision(sample,svmStruct);
                catch ME
                     error(message('stats:svmclassify:ClassifyFailed', ME.message));
                end             
            else
                predicted_class = [];
            end
        end
        
        
        function [predicted_class,score] = svmdecision(test_features,svm_struct)
            
            w = svm_struct.w;
            b = svm_struct.b;
            score = test_features * w + b;
            % score>0 --> 1
            % score=0 --> 0
            % score<0 --> -1
            predicted_class = sign(score);
            % points on the boundary are assigned to class 1
            predicted_class(predicted_class==0) = 1;
        end
        
        function [Model,predicted] = classify(Sample,class,SampleTest)
            Model=svm.train(Sample,class);
            predicted=svm.predict(Model,SampleTest);
        end
    end
end