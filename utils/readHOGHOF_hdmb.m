function [location, sigma2, tau2, descriptor ] = readHOGHOF_hdmb (filename)
% [ location, sigma2, tau2, descriptor ] = readHOGHOF (filename, descStr)
%
% Parameters:
%
%   descStr:        'hog', 'hof', 'hnf'
% For HDMB51 stip text files only
fid = fopen(filename);
location = zeros(5000000,3);
sigma2 = zeros(5000000,1);
tau2 = zeros(5000000,1);
descriptor = zeros(5000000,162);

count = 1;
while 1
    instr = fgets(fid);
    if instr == -1
        break;
    elseif instr(1) == '#'
        continue;
    else
        vec = str2num(instr);
        location(count,:) = [vec(2), vec(3), vec(4)]';
        sigma2(count) = vec(5);
        tau2(count) = vec(6);
        descriptor(count,:) = vec(8:end);
        count = count+1;
    end
end
disp(sprintf('read %d descriptors', count-1));
fclose(fid);

maxnum = find(sigma2 == 0, 1 );
location(maxnum:end,:) = [];
sigma2(maxnum:end) = [];
tau2(maxnum:end) = [];
descriptor(maxnum:end,:) = [];