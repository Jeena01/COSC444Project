% Remember to add paths for all folders and subfolders!
function test()

setup

det_pars.ubfpff_scale = 3;
det_pars.ubfpff_thresh = -0.75;
det_pars.iou_thresh = 0.9;

image = imread('test_images/img_main.jpg');
[ubfdetections] = DetectStillImage2(image, 'pff_model_upperbody_final.mat', 'haarcascade_frontalface_alt2.xml', det_pars, 2)
end