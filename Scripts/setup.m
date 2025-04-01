
base = pwd;
addpath(genpath(base))
try 
    vocPath = "Calvin detector/voc-release3.1/";
    calvinPath = "Calvin detector/calvin_upperbody_detector_v1.04/";
    poseEstimPath = "Calvin detector/pose_estimation_code_release_v1.22/code";
    esrganPath = "ESRGAN_Release100";

    addpath(vocPath)
    addpath(calvinPath)
    addpath(poseEstimPath)
    addpath(esrganPath)

    cd(poseEstimPath)
    installmex_2 
    cd(base)
    
    cd(calvinPath)
    load('detenv.mat')
    cd(base)

    % SKIP STEP, CALVIN DETECTOR NOT USED
    %cd(vocPath)
    %compile
    %cd(base)
catch e
    cd(base)
    sprintf("\n%s\n%s",e.identifier,e.message)
end
