function dln = ESRGAN_Generator()
    % 以下のコマンドでネットワークの構造を図示。
    % analyzeNetwork(ESRGAN_Generator());
        
    nResBlocks = 16;
    nFeatures = 64;
    heScale=3;
    leak=0.2;
    
    % input image pixel value range [0 1]
    % output image pixel value range [-1 1]
    
    %% Head.
    [lg, headLastName] = BuildHeadBlock(nFeatures, heScale, leak);
    prevOutName = headLastName;
    
    %% Residual block.
    for i=1:nResBlocks
        [lg, prevOutName] = BuildResidualBlock(nFeatures, lg, i, prevOutName, heScale, leak);
    end
    
    %% Tail.
    lg = BuildTail(nFeatures, lg, headLastName, prevOutName, heScale, leak);
    dln = dlnetwork(lg);
end

function [lg, tailName] = BuildHeadBlock(nFeatures, heScale, leak)
    nKernelSize = 3;
    
    %          RGB colorのとき [24 24 3]等にする。
    %        Y grayscaleのとき [24 24]等にする。
    inputL = imageInputLayer( [112 112 3], 'Name', 'Input', 'Normalization', 'none' );

    convHead = convolution2dLayer(nKernelSize*3, nFeatures, 'Padding', 'same', ...
            'WeightsInitializer', @(sz) leakyHe(sz,heScale), ...
            'Name', 'conv_Head');
    actHead = leakyReluLayer(leak, 'Name', 'LReLU_Head');

    lg = layerGraph( [ inputL convHead actHead ] );
    tailName = 'LReLU_Head';
end

function [lg, rbTailName] = BuildResidualBlock(nFeatures, lg, idx, prevOutName, heScale, leak)
    nKernelSize = 3;
    
    rbHeadName = strcat('conv_A', num2str(idx));
    
    convA = convolution2dLayer(nKernelSize, nFeatures, 'Padding', 'same', ...
            'WeightsInitializer', @(sz) leakyHe(sz,heScale), ...
            'Name', rbHeadName);
    actA = leakyReluLayer(leak, 'Name', strcat('LReLU_A', num2str(idx)));
    addA = additionLayer(2, 'Name', strcat('add_A', num2str(idx)));
    
    convB = convolution2dLayer(nKernelSize, nFeatures, 'Padding', 'same', ...
            'WeightsInitializer', @(sz) leakyHe(sz,heScale), ...
            'Name', strcat('conv_B', num2str(idx)));
    actB = leakyReluLayer(leak, 'Name', strcat('LReLU_B', num2str(idx)));
    addB = additionLayer(3, 'Name', strcat('add_B', num2str(idx)));

    convC = convolution2dLayer(nKernelSize, nFeatures, 'Padding', 'same', ...
            'WeightsInitializer', @(sz) leakyHe(sz,heScale), ...
            'Name', strcat('conv_C', num2str(idx)));
    actC = leakyReluLayer(leak, 'Name', strcat('LReLU_C', num2str(idx)));
    addC = additionLayer(4, 'Name', strcat('add_C', num2str(idx)));

    convD = convolution2dLayer(nKernelSize, nFeatures, 'Padding', 'same', ...
            'WeightsInitializer', @(sz) leakyHe(sz,heScale), ...
            'Name', strcat('conv_D', num2str(idx)));
    actD = leakyReluLayer(leak, 'Name', strcat('LReLU_D', num2str(idx)));
    addD = additionLayer(5, 'Name', strcat('add_D', num2str(idx)));
    
    convE = convolution2dLayer(nKernelSize, nFeatures, 'Padding', 'same', ...
            'WeightsInitializer', @(sz) leakyHe(sz,heScale), ...
            'Name',  strcat('conv_E', num2str(idx)));

    rbTailName = strcat('scaleBeta_E', num2str(idx));
    scaleBetaE = functionLayer(@(X) X * 0.2, 'Name', rbTailName);

    lg = addLayers(lg, [ convA actA addA convB actB addB convC actC addC convD actD addD convE scaleBetaE] );
    
    % 前回出力を今回入力と連結。
    lg = connectLayers(lg, prevOutName,                   rbHeadName);

    lg = connectLayers(lg, prevOutName,                   strcat('add_A', num2str(idx), '/in2' ));

    lg = connectLayers(lg, prevOutName,                   strcat('add_B', num2str(idx), '/in2' ));
    lg = connectLayers(lg, strcat('add_A', num2str(idx)), strcat('add_B', num2str(idx), '/in3' ));

    lg = connectLayers(lg, prevOutName,                   strcat('add_C', num2str(idx), '/in2' ));
    lg = connectLayers(lg, strcat('add_A', num2str(idx)), strcat('add_C', num2str(idx), '/in3' ));
    lg = connectLayers(lg, strcat('add_B', num2str(idx)), strcat('add_C', num2str(idx), '/in4' ));
    
    lg = connectLayers(lg, prevOutName,                   strcat('add_D', num2str(idx), '/in2' ));
    lg = connectLayers(lg, strcat('add_A', num2str(idx)), strcat('add_D', num2str(idx), '/in3' ));
    lg = connectLayers(lg, strcat('add_B', num2str(idx)), strcat('add_D', num2str(idx), '/in4' ));
    lg = connectLayers(lg, strcat('add_C', num2str(idx)), strcat('add_D', num2str(idx), '/in5' ));
end

function lg = BuildTail(nFeatures, lg, headLastName, prevOutName, heScale, leak)
    nKernelSize = 3;

    % ↓RGB 3ch出力の場合3。
    %  Y  1ch出力の場合1。
    nOutCh = 3;

    convA = convolution2dLayer(nKernelSize, nFeatures, 'Padding', 'same', ...
            'WeightsInitializer', @(sz) leakyHe(sz,heScale), ...
            'Name', 'conv_TailA');
    addA = additionLayer(2, 'Name', 'add_TailA');
    
    convB = convolution2dLayer(nKernelSize, nFeatures*4, 'Padding', 'same', ...
            'WeightsInitializer', @(sz) leakyHe(sz,heScale), ...
            'Name', 'conv_TailB');

    psB = depthToSpace2dLayer([2 2], 'Name', 'Ps_TailB', 'Mode', 'dcr');

    actB = leakyReluLayer(leak, 'Name', 'LReLU_TailB');

    convC = convolution2dLayer(9, nOutCh, 'Padding', 'same', ...
            'WeightsInitializer', @(sz) leakyHe(sz,heScale), ...
            'Name', 'conv_TailC');

    % 2倍拡大用。
    lg = addLayers(lg, [ convA addA convB psB actB convC ]);

    lg = connectLayers(lg, prevOutName, 'conv_TailA');
    lg = connectLayers(lg, headLastName, 'add_TailA/in2');
end

