nX=2;nY=2;nF=8;
X1=1:32;
X1=reshape(X1,[2,2,8])
Z=zeros(4,4,2);

Z(1:2:nX*2, 1:2:nY*2, 1:fix(nF/4)) = X1(1:nX, 1:nY, 1:4:nF);
Z(2:2:nX*2, 1:2:nY*2, 1:fix(nF/4)) = X1(1:nX, 1:nY, 2:4:nF);
Z(1:2:nX*2, 2:2:nY*2, 1:fix(nF/4)) = X1(1:nX, 1:nY, 3:4:nF);
Z(2:2:nX*2, 2:2:nY*2, 1:fix(nF/4)) = X1(1:nX, 1:nY, 4:4:nF);

Z

% ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

dLdZ=Z
dLdX1=zeros(2,2,8);

dLdX1(1:nX, 1:nY, 1:4:nF) = dLdZ(1:2:nX*2, 1:2:nY*2, 1:fix(nF/4));
dLdX1(1:nX, 1:nY, 2:4:nF) = dLdZ(2:2:nX*2, 1:2:nY*2, 1:fix(nF/4));
dLdX1(1:nX, 1:nY, 3:4:nF) = dLdZ(1:2:nX*2, 2:2:nY*2, 1:fix(nF/4));
dLdX1(1:nX, 1:nY, 4:4:nF) = dLdZ(2:2:nX*2, 2:2:nY*2, 1:fix(nF/4));

dLdX1

% ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

nX=2;nY=2;nF=8;nB=2;
X1=1:64;
X1=reshape(X1,[2,2,8,2])
Z=zeros(4,4,2,2);

Z(1:2:nX*2, 1:2:nY*2, 1:fix(nF/4),:) = X1(1:nX, 1:nY, 1:4:nF,:);
Z(2:2:nX*2, 1:2:nY*2, 1:fix(nF/4),:) = X1(1:nX, 1:nY, 2:4:nF,:);
Z(1:2:nX*2, 2:2:nY*2, 1:fix(nF/4),:) = X1(1:nX, 1:nY, 3:4:nF,:);
Z(2:2:nX*2, 2:2:nY*2, 1:fix(nF/4),:) = X1(1:nX, 1:nY, 4:4:nF,:);

Z

% ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

dLdZ=Z
dLdX1=zeros(2,2,8,2);

dLdX1(1:nX, 1:nY, 1:4:nF,:) = dLdZ(1:2:nX*2, 1:2:nY*2, 1:fix(nF/4),:);
dLdX1(1:nX, 1:nY, 2:4:nF,:) = dLdZ(2:2:nX*2, 1:2:nY*2, 1:fix(nF/4),:);
dLdX1(1:nX, 1:nY, 3:4:nF,:) = dLdZ(1:2:nX*2, 2:2:nY*2, 1:fix(nF/4),:);
dLdX1(1:nX, 1:nY, 4:4:nF,:) = dLdZ(2:2:nX*2, 2:2:nY*2, 1:fix(nF/4),:);

dLdX1

                