function hogFeatures = extractHOGFeatures(imgs)

hogFeatures = [];
for i=1:length(imgs)
    feat = features(im2double(imgs{i}),8);
    hogFeatures(i,:) = feat(:)';
end

end