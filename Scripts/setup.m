
base = pwd;
addpath(genpath(base))
try 
    vocPath = "Calvin detector/voc-release3.1/";
    calvinPath = "Calvin detector/calvin_upperbody_detector_v1.04/";
    poseEstimPath = "Calvin detector/pose_estimation_code_release_v1.22/code";

    addpath(vocPath)
    addpath(calvinPath)
    addpath(poseEstimPath)

    cd(poseEstimPath)
    %installmex_2 
    cd(base)
    
    cd(calvinPath)
    load('detenv.mat')
    cd(base)

    cd(vocPath)
    compile
    cd(base)
    
    cd(calvinPath)
    %[ubcdetections] = DetectStillImage('./example_data/images/000000.jpg','pff_model_upperbody_final.mat','haarcascade_frontalface_alt2.xml',det_pars,2)
    
    cd(base)
catch e
    cd(base)
    sprintf("\n%s\n%s",e.identifier,e.message)
end
