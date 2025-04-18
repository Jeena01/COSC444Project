mex -O resize.cc
mex -O dt.cc
mex -O features.cc

% use one of the following depending on your setup
% 1 is fastest, 3 is slowest 

% 1) multithreaded convolution using blas
%mex -O fconvblas.cc -lmwblas -output fconv
% 2) mulththreaded convolution without blas
% mex -O fconvMT.cc -output fconv
% 3) basic convolution, very compatible
mex -O fconv.cc -output fconv