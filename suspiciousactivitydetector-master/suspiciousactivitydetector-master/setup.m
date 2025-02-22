
path = pwd();
try 
    cd ./code/pose_estimation_code_release_v1.21/pose_estimation_code_release_v1.21/code;
    load("detenv.mat");
    disp("starting (wait)");
    startup

    cd ../../../voc-release3.1/voc-release3.1/;  

    fid1 = fopen('./resize.mexw64','r');
    fid2 = fopen('./dt.mexw64','r');
    fid3 = fopen('./features.mexw64','r');
    fid4 = fopen('./fconv.mexw64','r');
    if (fid1 > -1 && fid2 > -1 && fid3 > -1 && fid4 > -1) 
        disp("files already compiled");
        cd(path);
        return
    end

    addpath(genpath(pwd()))
    disp("compiling wait()");
    compile

    disp("setup");
    cd(path);
catch e
    fprintf("error occurred: %s",e.message);
    cd(path);
end