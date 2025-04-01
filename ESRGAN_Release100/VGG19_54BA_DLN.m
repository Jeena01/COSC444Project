% VGG19ネットワークに画像を入力し5_4層の特徴マップ出力(activation前)を取り出すディープラーニングネットワーク。
% ESRGANの論文に、Activation前のほうが結果が良いとある。
function dln = VGG19_54BA_DLN()
    v = vgg19;
    vL = v.Layers(:);

    % 画像のピクセル値のレンジをVGG19と合わせます。

    % 入力画像サイズを指定、Normalizationを無効にします。        
    inL = imageInputLayer([224 224 3], 'Normalization', 'none', 'Name', 'in_wo_normalize');
    
    % [-1 +1] → [-128 +128]
    scL = AddMulRGBLayer('scale_to_PM128', [0, 0, 0], [128.0, 128.0, 128.0]);
    
    dln = dlnetwork([inL scL vL(2:36)']);
end
