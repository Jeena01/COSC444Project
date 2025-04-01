clear;
clc;
clearvars;
close all;
rng('default');
rng(42);

startTick = tic;

if ~isfolder('trained')
    mkdir('trained');
end

%% 学習パラメーター。

trainName = 'RGB_Flickr2K_VGG54';

versionStr = '100';

% 2倍拡大のSuper ResolutionをTrainします。
% 4倍拡大に対応するためには、
% GeneratorのTailの拡大を2段階に改修、Discriminatorの層を増やしてinput layerを2倍のサイズに改修、
% VGG54のinput layerサイズを変更する必要があります。
imgScale = 2;

% input image size。GeneratorのinputLayerのサイズと合わせる。
% DiscriminatorのinputLayerサイズは、pathSize * imgScaleにする。
patchSize = [112 112]; 

% バッチサイズがギリギリすぎると、数時間後にメモリ不足になることがあるので若干余裕を持たせる。
miniBatchSz     = 12;

% 1つの画像からパッチを取得する数。
patchesPerImage = 2;

% Pre-trainのEpoch数。ESRGANモデルのPre-trainは5 epochsでは不足する感じだったので10にしました。
preTrainEpochs  = 10;

% pre-trainのlearning rate。0.001にすると発散。
preTrainLR = 0.0002;

% GAN-Trainのlearning rateはGANTrainNetwork()関数内にあります。

% Discriminatorを訓練するEpoch数。
preTrainDiscEpochs = 5;

% pre-trainが終わった後のgan trainのepoch数。
ganTrainEpochs  = 300; 

% Trainingを途中から再開する場合、ロードするpre-train済モデルepoch番号をセット。
% Pre-trainが終わって、GAN Trainを途中から再開する場合preTrainEpochsの番号をセットします。
% Pre-Train済モデルをロードしないとき(pre-trainを最初から実行)は0。
loadEpochPre = 0;

% GAN-Trainを途中から再開する場合、GAN trainのepoch番号をセット。
% GAN-Train済モデルをロードしないでGan-Trainを最初から開始するときは0。
loadEpochGAN = 0;

%% 訓練進捗プロット。

f = figure;
xywh=get(0,'ScreenSize');
plotSz=xywh(3)/4;
f.Position = [ 1 500 plotSz*4 plotSz ];

setappdata(gcf, 'SubplotDefaultAxesLocation', [0, 0, 1, 1]);
scoreAx = subplot(1,4,1);

xlabel("Iteration")
ylabel("Loss")

lineBlack = animatedline(scoreAx, 'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', 4, 'MaximumNumPoints', 100, 'Color', 'black');
lineBlue  = animatedline(scoreAx, 'LineStyle', 'none', 'Marker', 'x', 'MarkerSize', 4, 'MaximumNumPoints', 100, 'Color', 'b');
lineRed   = animatedline(scoreAx, 'LineStyle', 'none', 'Marker', '+', 'MarkerSize', 4, 'MaximumNumPoints', 100, 'Color', 'r');

hleg = legend('PreTrainGenerator', '-', '-', 'Location', 'southwest');

title('Reading train files...')

% 画像を貼る領域。
imgLRAx = subplot(1,4,2);
imgSRAx = subplot(1,4,3);
imgHRAx = subplot(1,4,4);

drawnow

%% ネットワーク。

dlnG = ESRGAN_Generator();
dlnD = SRGAN_Discriminator();

% VGG19 5_4 "before activation" layer.
dlnVGG = VGG19_54BA_DLN();

trailAvgG = [];
trailAvgSqG = [];
trailAvgD = [];
trailAvgSqD = [];


%% 学習データ読み込み。

% 学習データはあらかじめcreateTrainingSetAll_Combined.mで作成してください。
trainLRImgs = imageDatastore(['Flickr2kAll_RGB_MatlabF2' filesep 'train_' num2str(imgScale) 'x_small_mat'], 'FileExtensions','.mat','ReadFcn',@matRead);
trainHRImgs = imageDatastore(['Flickr2kAll_RGB_MatlabF2' filesep 'train_' num2str(imgScale) 'x_gt_mat'],    'FileExtensions','.mat','ReadFcn',@matRead);
nTrainImgs = numel(trainHRImgs.Files);

% Epochあたりite数。
nItePerEpoch = fix(nTrainImgs * patchesPerImage / miniBatchSz);

dsTrain = randomPatchSmallLargePairDataStore(trainLRImgs, trainHRImgs, patchSize, imgScale, ...
     'DataAugmentation', 'none', 'PatchesPerImage', patchesPerImage);
 
% MiniBatchFormatについて。
%
% imageInputLayer( [96 96 3], ... );の場合: RGB 3ch color
% 96x96x3x32 dlarray
%  S  S C B
%
% imageInputLayer( [96 96 1], ... );の場合: Y 1ch grayscale
% 96x96x32 dlarray
%  S  S B C
%
%    S — Spatial
%    C — Channel
%    B — Batch observations
mbqT = minibatchqueue(dsTrain, ...
    'MiniBatchSize', miniBatchSz, ...
    'MiniBatchFormat','SSCB',...
    'PartialMiniBatch', 'discard');

%% Pre-train。

if 0 < loadEpochPre
    % Pre-train学習済モデルを読み込む。
    fname = sprintf('trained/ESRGAN%s_preTrainG_%s_%dx_epoch%d.mat', versionStr, trainName, imgScale, loadEpochPre);
    load(fname);
end

if loadEpochPre < preTrainEpochs
    % pre-trainします。
    startEpoch = loadEpochPre + 1;
    dlnG = PreTrainNetwork(preTrainLR, trainName, imgScale, dlnG, ...
            imgLRAx, imgSRAx, imgHRAx, lineBlack, preTrainEpochs, trailAvgG, trailAvgSqG, startTick, startEpoch, nItePerEpoch, ...
            mbqT, versionStr);
end

%% Gen / Disc学習。

% pre-train後に続けてGen/Disc学習を行おうとすると OUT OF MEMORYエラーが起き、
% pre-trainのデータをロードして、Gen/Disc学習から始めるとエラーが起きないことがある。

if 0 < loadEpochGAN
    % Gen / Disc学習済モデルを読み込む。
    fname = sprintf('trained/ESRGAN%s_GANTrain_%s_%dx_epoch%d.mat', versionStr, trainName, imgScale, loadEpochGAN);
    load(fname);
end


clearpoints(lineBlack);

hleg.String = {'DiscHRImg', 'DiscSRImg', 'Discriminator'};

% GANでtrainします。

startEpochGAN = loadEpochGAN + 1;
dlnG = GANTrainNetwork(trainName, imgScale, dlnG, dlnD, dlnVGG, miniBatchSz, ...
        imgLRAx, imgSRAx, imgHRAx, lineBlack, lineBlue, lineRed, ganTrainEpochs, trailAvgG, ...
        trailAvgSqG, trailAvgD, trailAvgSqD, startTick, startEpochGAN, nItePerEpoch, preTrainDiscEpochs, ...
        mbqT, versionStr);

%% Pre-train : 通常のMSE loss Train。
function dlnG = PreTrainNetwork(preTrainLR, trainName, imgScale, dlnG, ...
        imgLRAx, imgSRAx, imgHRAx, lineBlack, preTrainEpochs, trailAvgG, trailAvgSqG, startTick, loadEpochPre, nItePerEpoch, ...
        mbqT, versionStr)
    for epoch = loadEpochPre : preTrainEpochs
        fprintf('PreTrainNetwork Epoch %d\n', epoch)

        shuffle(mbqT);
        clearpoints(lineBlack);

        ite = 0;
        while hasdata(mbqT)
            ite = ite + 1;
            
            [imgLR, imgHR] = next(mbqT);
            [gradG, lossG, imgSR] = dlfeval(@preTrainGen, dlnG, imgLR, imgHR);
            
            [dlnG,trailAvgG,trailAvgSqG] = adamupdate(dlnG, gradG, ...
                trailAvgG, trailAvgSqG, ite, preTrainLR);

            % Update the scores plot.
            lossValue = double(gather(extractdata(lossG)));
            subplot(1,4,1);
            addpoints(lineBlack, ite, lossValue);
            % Update the title with training progress information.
            D = duration(0,0,toc(startTick),'Format','hh:mm:ss');
            title(...
                "Pre-train Generator Epoch " + epoch + ...
                ", Ite " + ite + " / " + nItePerEpoch + ...
                ", " + string(D))

            if mod(ite, 10) == 0 || ite == 1
                % 10 iteに1度入出力画像表示更新。
                showImg(imgLR, imgSR, imgHR, imgLRAx, imgSRAx, imgHRAx);
            end
            
            drawnow
        end
        
        fname = sprintf('trained/ESRGAN%s_preTrainG_%s_%dx_epoch%d.mat', versionStr, trainName, imgScale, epoch);
        save(fname ,'dlnG', 'trailAvgG', 'trailAvgSqG');
    end
end

%% GANを使用したTrain。画像が発散する場合learning rateを下げて下さい。
function dlnG = GANTrainNetwork(trainName, imgScale, dlnG, dlnD, dlnVGG, miniBatchSz, ...
        imgLRAx, imgSRAx, imgHRAx, lineBlack, lineBlue, lineRed, ganTrainEpochs, trailAvgG, ...
        trailAvgSqG, trailAvgD, trailAvgSqD, startTick, startEpochGAN, nItePerEpoch, preTrainDiscEpochs, ...
        mbqT, versionStr)
    % 初期Learning rate。
    lrG = 0.0002;
    lrD = 0.00015;

    % lrDecayPeriod epoch毎にLRを半分にします。
    lrDecayPeriod = 100;    
    
    trainGenDisc = 0;

    subplot(1,4,1)
    ylim([0 1])
    xlabel("Iteration")
    ylabel("Score")

    for epoch = 1 : ganTrainEpochs
        if rem(epoch, lrDecayPeriod) == 0
            % learning rateを減らします。
            lrD = lrD / 2;
            lrG = lrG / 2;
        end        
        
        if epoch < startEpochGAN
            % スキップします。
        else
            fprintf('GANTrainNetwork Epoch=%d lrG=%f lrD=%f\n', epoch, lrG, lrD)

            if epoch <= preTrainDiscEpochs
                % あらかじめDiscriminatorをTrainする。
                mode = "Train Gen / Disc separately";
            else
                % GAN学習開始。
                mode = "Train Gen-Disc connected";
                trainGenDisc = 1;
            end

            shuffle(mbqT);
            clearpoints(lineBlack);
            clearpoints(lineBlue);
            clearpoints(lineRed);

            ite = 0;
            while hasdata(mbqT)
                ite = ite + 1;

                % ミニバッチを取得。
                [imgLR, imgHR] = next(mbqT);

                [gradG, gradD, imgSR, scoreRR, scoreFR, scoreDisc] ...
                    = dlfeval(@ganTrainGenDisc, dlnG, dlnD, dlnVGG, imgLR, imgHR, miniBatchSz, trainGenDisc);

                % 係数更新。
                [dlnG,trailAvgG,trailAvgSqG] = adamupdate(dlnG, gradG, ...
                    trailAvgG, trailAvgSqG, ite, lrG);

                [dlnD, trailAvgD, trailAvgSqD] = adamupdate(dlnD, gradD, ...
                    trailAvgD, trailAvgSqD, ite, lrD);

                % グラフ更新。
                subplot(1,4,1)
                scoreRR2 = double(gather(extractdata(scoreRR)));
                addpoints(lineBlack, ite, scoreRR2);

                scoreFR2 = double(gather(extractdata(scoreFR)));
                addpoints(lineBlue, ite, scoreFR2);

                scoreDisc2 = double(gather(extractdata(scoreDisc)));
                addpoints(lineRed, ite, scoreDisc2);

                % タイトル文字列更新。
                subplot(1,4,1);
                D = duration(0,0,toc(startTick),'Format','hh:mm:ss');
                title(...
                    mode + " Ep " + epoch + ...
                    ", Ite " + ite + " / " + nItePerEpoch + ...
                    ", " + string(D))

                if mod(ite, 10) == 0 || ite == 1
                    showImg(imgLR, imgSR, imgHR, imgLRAx, imgSRAx, imgHRAx);
                end

                drawnow
            end

            % 進捗をファイルに保存。
            fname = sprintf('trained/ESRGAN%s_GANTrain_%s_%dx_epoch%d.mat', versionStr, trainName, imgScale, epoch);
            save(fname ,'dlnG', 'dlnD',  'trailAvgG', 'trailAvgSqG', 'trailAvgD', 'trailAvgSqD');

            fname = sprintf('trained/ESRGAN%s_%s_%dx_Generator_params_epoch%d.mat', versionStr, trainName, imgScale, epoch);
            save(fname ,'dlnG');
        end
    end
end

%% サブルーチン群。

% preTrainのdlfevalから呼び出される。
function [gradG, lossG, imgSR] = preTrainGen(dlnG, imgLR, imgHR)
    [imgSR, ~] = forward(dlnG, imgLR);

    lossG = mse(sigmoid(imgHR), sigmoid(imgSR));

    gradG = dlgradient(lossG, dlnG.Learnables, 'EnableHigherDerivatives', false);
end

% GANTrainNetworkのdlfevalから呼び出される。
function [gradG, gradD, imgSR, scoreRR, scoreFR, scoreD] = ...
ganTrainGenDisc(dlnG, dlnD, dlnVGG, imgLR, imgHR, miniBatchSz, trainGenDisc)
    [imgSR, ~] = forward(dlnG, imgLR);
    % ESRGAN paper pp. 6 to 7

    % GT画像のDiscriminator output。
    cXr = forward(dlnD, imgHR);
    
    % Gen画像のDiscriminator output。
    cXf = forward(dlnD, imgSR);
    
    % GT画像をDiscに入力、Realと判定されたら1、Fakeと判定されたら0。
    dXr = sigmoid(cXr);

    % Gen画像をDiscに入力、Realと判定されたら1、Fakeと判定されたら0。
    dXf = sigmoid(cXf);
    
    % Discのスコア。GT画像が全て１と判定され、Gen画像が全て0と判定されると最大値1。
    scoreD = (mean(dXr) + mean(1- dXf)) * 0.5;

    % Real画像がすべて1と判定されたら1。
    scoreRR = mean(dXr);
    
    % Genのスコア。Gen画像がすべて1と判定されたら最大値1。
    scoreFR = mean(dXf);

    % ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    % discriminator loss is binary cross-entropy loss, 
    % on Matlab, cross-entropy loss for multi-label classification is crossentropy('TargetCategories', 'independent')
    % ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    tgt_1 = dlarray( ones(1, 1, 1, miniBatchSz, 'single'), 'SSCB');
    tgt_0 = dlarray(zeros(1, 1, 1, miniBatchSz, 'single'), 'SSCB');

    % D_{Ra}(x_r, x_f)
    dXrf = sigmoid(cXr - mean(cXf));

    % D_{Ra}(x_f, x_r)
    dXfr = sigmoid(cXf - mean(cXr));

    lossRF1 = crossentropy(dXrf, tgt_1, 'TargetCategories', 'independent');
    lossFR0 = crossentropy(dXfr, tgt_0, 'TargetCategories', 'independent');
    lossD = (lossFR0 + lossRF1) * 0.5;

    lossGenMSE = mse(sigmoid(imgHR), sigmoid(imgSR));

    if trainGenDisc == 0
        lossG = lossGenMSE;
        fprintf('lG=%f lD=%f\n', lossG, lossD);
    else
        % Generator loss from Discriminator
        lossRF0 = crossentropy(dXrf, tgt_0, 'TargetCategories', 'independent');
        lossFR1 = crossentropy(dXfr, tgt_1, 'TargetCategories', 'independent');
        lossGenFromDisc = (lossFR1 + lossRF0) * 0.5;

        % VGG19_54BAロスを計算する。
        featGT  = runVGG19_54BA(dlnVGG, imgHR);
        featGen = runVGG19_54BA(dlnVGG, imgSR);
        lossGenContent = mse(sigmoid(featGT), sigmoid(featGen));

        % TEST: VGG54。画像がやや白っぽくなる。
        %lossG =                (1.0/1.0e3) * lossGenContent;
        %fprintf('lVGG54=%f\n', (1.0/1.0e3) * lossGenContent);

        % TEST: MSE + VGG54。
        %lossG =                        (1.0/7.0) * lossGenMSE +(1.0/1.0e3) * lossGenContent;
        %fprintf('lMSE=%f lVGG54=%f\n', (1.0/7.0) * lossGenMSE, (1.0/1.0e3) * lossGenContent);

        % TEST: MSE + GANロス。
        %lossG =                      (1.0/7.0) * lossGenMSE +(1.0/1.0) * lossGenFromDisc;
        %fprintf('lMSE=%f lGAN=%f\n', (1.0/7.0) * lossGenMSE, (1.0/1.0) * lossGenFromDisc);
        
        % 各lossの値の大きさを比べ、大体同じくらいのバランス、MSEやや強めにしたもの。
        lossG =                                (3.0/7.0) * lossGenMSE + (3.0/5.0) * lossGenFromDisc + (3.0/1.0e3) * lossGenContent;
        fprintf('lMSE=%f lGAN=%f lVGG54=%f\n', (3.0/7.0) * lossGenMSE,  (3.0/5.0) * lossGenFromDisc,  (3.0/1.0e3) * lossGenContent);
    end

    gradG = dlgradient(lossG, dlnG.Learnables, 'EnableHigherDerivatives', false);
    gradD = dlgradient(lossD, dlnD.Learnables, 'EnableHigherDerivatives', false);
end

% -------------------------------------------------------------------------------------------------

%% VGG19の5_4層目から特徴量を得ます。
function feat54 = runVGG19_54BA(dlnVGG, dlImg)
    % VGG19から5_4層の出力(before activation)を取り出す。
    feat54 = forward(dlnVGG, dlImg);
end

%% 画像を何枚か取り出してタイル状に並べます。
function I = convDLNtoImg(dln, isLR)
    I = extractdata(dln);    
    
    % 表示領域に収まるよう画像の枚数を減らす。
    % 第4引数 1:4 → 1番目から4枚取り出す。
    % 第4引数 1 → 1番目を取り出す。
    %     S S C  B
    I = I(:,:,:, 1);
    
    if isLR == 1
        % LR画像は[0 1]レンジ。
    else
        % SR,HR画像は[-1 +1]レンジなので、[0 1]に変換する。
        I = I * 0.5 + 0.5;
    end
    
    % Y grayscaleの場合: RGB 3chに変換する。
    % I = cat(3, I,I,I);
    
    % RGB用の処理。
    szI = size(I);
    if numel(szI) == 3
        % 1枚のみ選択の場合。Iは既に画像として表示できる。
        I = imresize(I, 3, 'nearest');
    else
        % タイル状に並べます。
        I = imtile(I);
    end
    
end

function showImg(imgLR, imgSR, imgHR, imgLRAx, imgSRAx, imgHRAx)
    subplot(1,4,2)
    I = convDLNtoImg(imgLR, 1);
    image(imgLRAx,I);
    axis(imgLRAx, 'image');
    set(gca,'xtick',[],'ytick',[]);
    title("Input Low Res img (2x downsampled from High Res)");

    subplot(1,4,3)
    I = convDLNtoImg(imgSR, 0);
    image(imgSRAx,I);
    axis(imgSRAx, 'image');
    set(gca,'xtick',[],'ytick',[]);
    title("ESRGAN Super Res Output img");

    subplot(1,4,4)
    I = convDLNtoImg(imgHR, 0);
    image(imgHRAx,I);
    axis(imgHRAx, 'image');
    set(gca,'xtick',[],'ytick',[]);
    title("Original High Res img");
end

