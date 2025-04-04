% https://www.mathworks.com/help/deeplearning/ug/specify-custom-weight-initialization-function.html
function weights = leakyHe(sz,scale)

% If not specified, then use default scale = 0.1
if nargin < 2
    scale = 0.1;
end

filterSize = [sz(1) sz(2)];
numChannels = sz(3);
numIn = filterSize(1) * filterSize(2) * numChannels;

varWeights = 2 / ((1 + scale^2) * numIn);
weights = randn(sz) * sqrt(varWeights);

end