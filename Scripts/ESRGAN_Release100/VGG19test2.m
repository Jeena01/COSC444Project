% VGG19モデルで訓練画像を分類するVGG19の動作テストプログラム。

% Load the trained model
netVGG19 = vgg19()
netVGG19.Layers

% 画像のピクセル値のレンジをVGG19と合わせます。

vL = netVGG19.Layers(:);

% 入力画像サイズを指定、Normalizationを無効にします。
inL = imageInputLayer([224 224 3], 'Normalization', 'none');

% [-1 +1] → [-128 +128]
scL = AddMulRGBLayer('scale_M128_to_128', [0, 0, 0], [128.0, 128.0, 128.0]);
net = SeriesNetwork([inL scL vL(2:end)']);    
        

%% 学習データ読み出し。

% 学習データはあらかじめcreateTrainingSetAll_Flickr2K.mで作成してください。
patchSize = [112 112]; % input image size
imgScale = 2;
miniBatchSz =     8 * 8;
patchesPerImage = 8 * 8;

trainSmallImgs = imageDatastore(['Combined_RGB_MatlabF2' filesep 'train_' num2str(imgScale) 'x_small_mat'], 'FileExtensions','.mat','ReadFcn',@matRead);
trainOrigImgs  = imageDatastore(['Combined_RGB_MatlabF2' filesep 'train_' num2str(imgScale) 'x_gt_mat'],    'FileExtensions','.mat','ReadFcn',@matRead);
nTrainImgs = numel(trainOrigImgs.Files);

dsTrain = randomPatchSmallLargePairDataStore(trainSmallImgs, trainOrigImgs, patchSize, imgScale, ...
     'DataAugmentation', 'none', 'PatchesPerImage', patchesPerImage);

mbqT = minibatchqueue(dsTrain, ...
    'MiniBatchSize', miniBatchSz, ...
    'MiniBatchFormat','SSCB',...
    'PartialMiniBatch', 'discard');

count = 0;
while hasdata(mbqT)
    count = count + 1;
    [~, dlImg] = next(mbqT);

    img = extractdata(dlImg);
    label = classify(net, img)
end

