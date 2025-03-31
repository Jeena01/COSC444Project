
addpath("./utils/")
cd("utils/")

mex -O mexDGC.cpp
mex -O nema_lognorm_fast.cxx

mex -O vgg_nearest_neighbour_dist.cxx
mex -O triLinearVoting.cpp
mex -O triLinearInterpolation.cpp
%mex -O matlabPorts.h
%mex -O graph.h
%mex -O block.h