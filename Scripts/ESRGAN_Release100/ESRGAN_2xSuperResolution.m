function Isr = ESRGAN_2xSuperResolution(Ilr)
    scale = 2;

    % SRGAN trained network ==> dlnG
    load('trained/ESRGAN100_RGB_Flickr2K_VGG54_2x_Generator_params_epoch300.mat');

    % Ilrを超解像しIsrを作る。
    
    Ilr_s = im2single(Ilr);
    Ilr_dl = dlarray(Ilr_s, 'SSCB');
    
    [Isr_dl, stateG] = forward(dlnG, Ilr_dl);
    
    Isr = single(extractdata(Isr_dl));
    Isr = Isr * 0.5 + 0.5;    
    
    %figure;
    %imshow(Isr);
    %title('ESRGAN Super Resolution Image');
end





