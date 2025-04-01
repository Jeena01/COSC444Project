function calvin_test(varargin)
    close all;
    setup;
    
    load("Calvin detector/pose_estimation_code_release_v1.22/code/env.mat");
    load("Calvin detector/calvin_upperbody_detector_v1.04/code/detenv.mat");
    pic = 'rock_climber.jpg';
    if (nargin > 0)
        pic = varargin{1};
    end

    DetectAndEstim('Images/',pic,'Calvin detector/calvin_upperbody_detector_v1.04/code/pff_model_upperbody_final.mat',[],det_pars,'full',fghigh_params,parse_params_Buffy3and4andPascal,[],pm2segms_params,1);
end