function [idt] = readIDT (filename)
% Parameters:
%
%   descStr:  'trajectory (30)', 'hog(96)', 'hof(108)', 'mbh(192)'
%   Total: 426 Dimension

fid = fopen(filename);
idt = zeros(500000,426);

count = 1;
while 1
    instr = fgets(fid);
    if instr == -1
        break;
    elseif instr(1) == '#'
        continue;
    else
        vec = str2num(instr);
        idt(count,:) = vec(11:end);
        count = count+1;
    end
end
fprintf('read %d descriptors\n', count-1);
fclose(fid);

idt(count:end,:) = [];
