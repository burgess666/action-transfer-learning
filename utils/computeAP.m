function [ AP , tp] = computeAP( scores,gt,topN)

%{
    computes Average Precision for the given scores and ground truth

if (nargin==2)
    topN = length(scores);
end

maxTp = sum(gt==1);

sorted = sortrows([scores gt],-1);
sumprec = 0;
tp = 0;
AP=0;
for i=1:topN
    if(sorted(i,2)==1)
        tp=tp+1;
        sumprec = sumprec +  double(tp)/i;
    end
end

if (tp~=0)
    AP = double(sumprec) / maxTp;
end

%}

% calculate accuracy
if (nargin==2)
    topN = length(scores);
end

maxTp = sum(gt==1);

sorted = sortrows([scores gt],-1);
sumprec = 0;
tp = 0;
AP=0;
for i=1:topN
    if(sorted(i,2)==1)
        tp=tp+1;
        sumprec = sumprec +  double(tp)/i;
    end
end

if (tp~=0)
    AP = double(sumprec) / maxTp;
end