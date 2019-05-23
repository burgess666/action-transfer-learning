function drawComparisonFigure(mytitle,sampling,all,errbar)

if nargin <4
    errbar = 0;
end
    
styles = { ...
    '-o', 
    '--x',    
    '--s',
    '--v',
    '--', 
    '.--',
    '-', 
    '--'};

colors = [ ...
    'r';        
    'g';
    'b';
    'c';        
    'm';        
    'y';
    'k'
    ];

% figure(1);
clf;
hold on

xt = size(all{1,1});

for i=1:size(all,1)    
    if (size(all{i,1},1) == 1)
        all{i,1} = [all{i,1};all{i,1}];
    end
end

for i=1:size(all,1)    
    if (size(all{i,1},2)>1)
        L = mean(all{i,1})-min(all{i,1});
        U = max(all{i,1})-mean(all{i,1});
        plot(sampling(:),mean(all{i,1}), styles{i} ,'LineWidth', 1 ,'Color',colors(i,:));
        if errbar    
        errorbar(sampling(:),mean(all{i,1}), L , U ,styles{i} ,'LineWidth', 2 ,'Color',colors(i,:));
        end
    else
        plot(sampling(:),all{i,1}, styles{i} ,'LineWidth', 1 ,'Color',colors(i,:));
    end
    
end
h = legend(all(:,2),'Location','SouthEast','FontSize', 14);

xlabel('Percentage of Target Samples', 'FontSize', 16, 'fontweight','b')
ylabel('Average Recall', 'FontSize', 16, 'fontweight','b')

set(gca,'XTick',sampling, 'FontSize', 14 ,'fontweight','b') %[1 5 10 15 20]
set(gca,'YTick',0:0.1:1, 'FontSize', 14 ,'fontweight','b')
set(gca,'YGrid','on');
set(gca,'XGrid','on');
title(mytitle,'FontSize', 18, 'fontweight','b')
%axis([1 max(sampling) 0.6 1]);

