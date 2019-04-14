function [accuracy] = calCM(trueMat, predictedMat)
% This function calculates True Positives, False Positives, True Negatives
% and False Negatives for two matrices of equal size assuming they are
% populated by 1's and -1's.
% The trueMat contains the actual true values while the predictedMat
% contains the 1's and -1's predicted from the algorithm used.

predicted = zeros(length(predictedMat),1)';

for i = 1:length(predictedMat)
    if predictedMat(i) >= 0
        predicted(i) = 1;
    else
        predicted(i) = -1;
    end
end

adder = trueMat + predicted;
TP = length(find(adder == 2));
TN = length(find(adder == -2));
subtr = trueMat - predicted;
FP = length(find(subtr == -2));
FN = length(find(subtr == 2));
fprintf('TP = %d \n TN = %d \n FP = %d \n FN = %d\n',TP,TN,FP,FN);
%precision = TP/(TP+FP);
%fprintf('Precision: %d',precision)
accuracy = (TP+TN)/(TP+TN+FP+FN);
fprintf('\nAccuracy: %d \n',accuracy)

% below codes for 1 and 0 situation
%{
adder = trueMat + predictedMat;
TP = length(find(adder == 2));
TN = length(find(adder == 0));
subtr = trueMat - predictedMat;
FP = length(find(subtr == -1));
FN = length(find(subtr == 1));
%}