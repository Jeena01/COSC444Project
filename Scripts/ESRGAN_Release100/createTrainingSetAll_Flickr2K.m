imagesDir='.';

% Flickr2K/Flickr2K_HRディレクトリ:オリジナルの鮮明な画像の置き場。Flickr2KデータセットのPNG画像を置きます。
trainImagesDir=fullfile(imagesDir,'Flickr2K' , 'Flickr2K_HR');

exts={'.png'};

trainPristineImages=imageDatastore(trainImagesDir,'FileExtensions',exts);

% numel(trainPristineImages.Files)

% ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

% mean and std results of this dataset:
% meanRGB = (0.480534 0.452418 0.407630)
% stdRGB  = (0.235300 0.225423 0.235360)

outDir = 'Flickr2k_RGB_MatlabF2';
if ~isfolder(outDir)
    mkdir(outDir);
end

for scale = [2]

    trainGtPngDirName  = [outDir filesep sprintf('train_%dx_gt_png',scale) ];
    trainGtDirName     = [outDir filesep sprintf('train_%dx_gt_mat',scale) ];
    trainSmallDirName  = [outDir filesep sprintf('train_%dx_small_mat', scale) ];

    createTrainingSet2(trainPristineImages, scale, trainGtDirName, trainGtPngDirName, trainSmallDirName);
end


