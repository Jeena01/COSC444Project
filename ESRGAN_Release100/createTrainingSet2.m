function createTrainingSet2(imds, scale, hrImgMinSize, hrImageDirName, halfImageDirName, smallImgDirName)
    if ~isfolder(hrImageDirName)
        mkdir(hrImageDirName);
    end
    
    if ~isfolder(halfImageDirName)
        mkdir(halfImageDirName);
    end
    
    if ~isfolder(smallImgDirName)
        mkdir(smallImgDirName);
    end

    c=1;
    
    accStd = [0.0, 0.0, 0.0];
    accMean = [0.0, 0.0, 0.0];
    
    nImg=numel(imds.Files);
    nImgProcessed = 0;
    
    reset(imds);
    while hasdata(imds)
        [I,info] = read(imds);
        [~,fileName,~] = fileparts(info.Filename);
        
        % まず半分にする。画像をくっきりさせるため。
        I = imresize(I, 0.5, 'Method', 'lanczos3');
        
        % グレースケールのY成分だけ使用する場合以下のようにする。
        %I = rgb2ycbcr(I);
        %I = I(:,:,1);

        I = im2single(I);    % ←floatにする。
        %I = im2double(Y);    % ←doubleにする。
        
        szGT = size(I);
        if szGT(1) < hrImgMinSize(1) || szGT(2) < hrImgMinSize(2)
            fprintf('image too small: %s\n', fileName)
            c = c + 1;
            continue;
        end

        % image size adjust to multiply of scale
        newSz = [int32((szGT(1:2)-(scale-1))/scale) * scale, 3];
        rect = images.spatialref.Rectangle([1 newSz(2)], [1 newSz(1)]);
        I = imcrop(I, rect);
        
        ILR = imresize(I, 1.0/scale, 'lanczos3');
        [nrowsL, ncolsL, ~] = size(ILR);

        imwrite(I, sprintf('%s/%s.png', halfImageDirName, fileName));
        
        % ILR: LR画像のレンジは[0 1]
        % I: HR画像のレンジは[-1 +1]
        I = I * 2.0 - 1.0;

        save([hrImageDirName    filesep fileName '.mat'],'I');
        save([smallImgDirName   filesep fileName '.mat'],'ILR');

        % 平均と標準偏差を算出。
        ILR_R = ILR(:,:,1);
        ILR_G = ILR(:,:,2);
        ILR_B = ILR(:,:,3);
        
        accStd(1) = accStd(1) + std(ILR_R(:));
        accStd(2) = accStd(2) + std(ILR_G(:));
        accStd(3) = accStd(3) + std(ILR_B(:));
        accMean(1) = accMean(1) + mean(ILR_R(:));
        accMean(2) = accMean(2) + mean(ILR_G(:));
        accMean(3) = accMean(3) + mean(ILR_B(:));
        nImgProcessed = nImgProcessed + 1;

        fprintf('%d / %d %dx%d\n', c, nImg, szGT(2), szGT(1));
        
        c = c + 1;
    end
    
    fprintf('nImgProcessed=%d std=%f %f %f mean=%f %f %f\n', ...
        nImgProcessed, accStd(1)/nImgProcessed, accStd(2)/nImgProcessed, accStd(3)/nImgProcessed , ...
        accMean(1)/nImgProcessed, accMean(2)/nImgProcessed, accMean(3)/nImgProcessed);    
end

