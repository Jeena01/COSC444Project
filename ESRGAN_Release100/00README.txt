ESRGAN Single Image Super Resolution Matlab port version 1.0.0.

■ Prerequisites ■

Matlab 2022a
Image Processing toolbox
Statistics and Machine Learning toolbox
Deep Learning Toolbox
Parallel Computing Toolbox

■ How to Test ■

Run ESRGAN_Test.m which calls ESRGAN_2xSuperResolution.m
Trained net is loaded on the line 5 of ESRGAN_2xSuperResolution.m

■ How to Perform ESRGAN Super-Resolution to your image file ■

Input image MyPicture.jpg should be pristine (not blurred) image. SRGAN neural net will upscale the image by 2x.

img = imread("MyPicture.jpg");          % 1024x768 input image 
imgSR = ESRGAN_2xSuperResolution(img);
imwrite(imgSR, "MyPicture_2x_SRGAN_MSE.png"); % 2048x1536 image is outputted

■ How to Train the network ■

Download Flickr2K dataset and place all png files on Flickr2K/Flickr2K_HR.
Run createTrainingSetAll_Flickr2K.m to create Flickr2K_RGB_MatlabF2 folder that contains converted mat files.
Run ESRGAN_Train.m to train and create trained model file.
Specify your trained model file on ESRGAN_2xSuperResolution.m to perform super resolution.

■ Difference from the original ESRGAN ■

1. Training low-resolution input image size is 112x112.
2. Flickr2K dataset is used to train the model.
3. Only 2x super resolution is implemented.
4. VGG19_54 loss, MSE loss, and GAN loss weighting ratio for Generator training is different.
5. MSE loss instead of MAE loss.

■ My training result becomes complete white image. How to fix it ■

・Reduce the learning rate.
・Run ESRGAN_Train.m and watch values of lossGenMSE, lossGenFromDisc, lossGenVGG54 on Command Window.
  If one value is significantly larger than other two, decrease it.

■ How to get more crisp image ■

Decrease lossGenMSE contribution of ESRGAN_Train.m:399 to get more crisp image. But artifact increases.

■ Changelog ■

Version 1.0.0

・Initial release.

■ References ■

Xintao Wang, et al. ESRGAN: Enhanced super-resolution generative adversarial networks. In ECCVW, 2018.
https://arxiv.org/abs/1809.00219

Ledig, C., Theis, L., Husz ́ar, F., Caballero, J., Cunningham, A., Acosta, A., Aitken,A., Tejani, A., Totz, J., Wang, Z., et al.: Photo-realistic single image super-resolution using a generative adversarial network. In: CVPR (2017)
https://arxiv.org/pdf/1609.04802.pdf

Single Image Super-Resolution Using Deep Learning
(VDSR is implemented using Matlab Deep Learning Toolbox)
https://www.mathworks.com/help/images/single-image-super-resolution-using-deep-learning.html

Train Generative Adversarial Network (GAN) using Matlab
https://www.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html

Monitor GAN Training Progress and Identify Common Failure Modes
https://www.mathworks.com/help/deeplearning/ug/monitor-gan-training-progress-and-identify-common-failure-modes.html

VGG-19 convolutional neural network (Matlab)
https://www.mathworks.com/help/deeplearning/ref/vgg19.html?searchHighlight=VGG19&s_tid=srchtitle


