clear;
clc;
clearvars;
close all;

% Scaling factor, how much to scale up the image
n = 2.0;

exts = {'.jpg','.png','.tif'};
fileNames = {'rock_climber.jpg'};

% filePath = [fullfile(matlabroot,'toolbox','images','imdata') filesep];
filePath = "D:\HAMZA\1.UBC Ok\COSC_O 544 Computer Vision\Project\UIUC Sports Event Dataset_RockClimbing\test\RockClimbing\";
filePathNames = strcat(filePath,fileNames);
testImages = imageDatastore(filePathNames,'FileExtensions',exts);

for indx = 1:numel(fileNames)
    fprintf('%d / %d\n', indx, numel(fileNames));
    
    % Future work, do object detection to find the human, draw bounding
    % box around him, get box coordinates, and crop to ROI for faster SR,
    % more efficient.
    % Read and preprocess the image
    Ireference = readimage(testImages, indx);    
    IrefC = im2single(Ireference);

    % Super-resolve using ESRGAN
    IsisrC = ESRGAN_2xSuperResolution(IrefC); % IlowresC

    % Convert to uint8 (scale back to [0,255])
    IsisrC_uint8 = im2uint8(IsisrC);
    
    % Extract the original filename (without extension)
    [~, name, ~] = fileparts(fileNames{indx});
    
    % filepath to save the super resolved image in
    save_folder = "D:\HAMZA\1.UBC Ok\COSC_O 544 Computer Vision\Project\Project_Codes\ESRGAN_Release100\SuperResolvedImages\";
    output_filename = fullfile(save_folder, sprintf('%s_SR.png', name));
    
    % Save the image with the new name
    imwrite(IsisrC_uint8, output_filename);
    
    figure(1);
    subplot(1,2,1);
    imshow(IrefC);
    title('Reference Image');

    subplot(1,2,2);
    imshow(IsisrC);
    title('ESRGAN Super-Resolution');
end
