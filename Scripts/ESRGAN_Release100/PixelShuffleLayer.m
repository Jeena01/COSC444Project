classdef PixelShuffleLayer < nnet.layer.Layer

    properties (Learnable)
    end
    
    methods
        function layer = PixelShuffleLayer(name) 
            layer.NumInputs = 1;
            layer.Name = name;

            layer.Description = "PixelShuffle";
        end
                
        function Z = predict(layer, X1)
            sz = size(X1);
            
            % 48x48x256 → 96x96x64にする。(一例)
            nX = sz(1); % 48
            nY = sz(2); % 48
            nF = sz(3); % 256
            
            if numel(sz) == 3
                Z = zeros( nX*2, nY*2, fix(nF / 4), 'like', X1 );
                
                % 下記の低速コードを高速化したつもり。
                Z(1:2:nX*2, 1:2:nY*2, 1:fix(nF/4)) = X1(1:nX, 1:nY, 1:4:nF);
                Z(2:2:nX*2, 1:2:nY*2, 1:fix(nF/4)) = X1(1:nX, 1:nY, 2:4:nF);
                Z(1:2:nX*2, 2:2:nY*2, 1:fix(nF/4)) = X1(1:nX, 1:nY, 3:4:nF);
                Z(2:2:nX*2, 2:2:nY*2, 1:fix(nF/4)) = X1(1:nX, 1:nY, 4:4:nF);
                
%                 % 上記コードと同じことをする低速なコード。            
%                 for x=1:nX
%                     for y=1:nY
%                         for f=1:fix(nF/4)
%                             Z(x*2-1, y*2-1, f) = X1(x, y, f*4-3);
%                             Z(x*2-0, y*2-1, f) = X1(x, y, f*4-2);
%                             Z(x*2-1, y*2-1, f) = X1(x, y, f*4-1);
%                             Z(x*2-0, y*2-1, f) = X1(x, y, f*4-0);
%                         end
%                     end
%                 end
            elseif numel(sz) == 4
                nB = sz(4);
                Z = zeros( nX*2, nY*2, fix(nF / 4), nB, 'like', X1 );

                Z(1:2:nX*2, 1:2:nY*2, 1:fix(nF/4),:) = X1(1:nX, 1:nY, 1:4:nF,:);
                Z(2:2:nX*2, 1:2:nY*2, 1:fix(nF/4),:) = X1(1:nX, 1:nY, 2:4:nF,:);
                Z(1:2:nX*2, 2:2:nY*2, 1:fix(nF/4),:) = X1(1:nX, 1:nY, 3:4:nF,:);
                Z(2:2:nX*2, 2:2:nY*2, 1:fix(nF/4),:) = X1(1:nX, 1:nY, 4:4:nF,:);
            end
        end
                
        function Z = forward(layer, X1)
            Z = predict(layer,X1);
        end

%         function dLdX1 = backward(layer, X1, Z, dLdZ, memory)
%             sz = size(X1);
%             dLdX1 = zeros( sz, 'like', X1 );
%             
%             nX = sz(1); % 48
%             nY = sz(2); % 48
%             nF = sz(3); % 256
%             
%             if numel(sz) == 3
%                 dLdX1(1:nX, 1:nY, 1:4:nF) = dLdZ(1:2:nX*2, 1:2:nY*2, 1:fix(nF/4));
%                 dLdX1(1:nX, 1:nY, 2:4:nF) = dLdZ(2:2:nX*2, 1:2:nY*2, 1:fix(nF/4));
%                 dLdX1(1:nX, 1:nY, 3:4:nF) = dLdZ(1:2:nX*2, 2:2:nY*2, 1:fix(nF/4));
%                 dLdX1(1:nX, 1:nY, 4:4:nF) = dLdZ(2:2:nX*2, 2:2:nY*2, 1:fix(nF/4));
%             elseif numel(sz) == 4
%                 dLdX1(1:nX, 1:nY, 1:4:nF,:) = dLdZ(1:2:nX*2, 1:2:nY*2, 1:fix(nF/4),:);
%                 dLdX1(1:nX, 1:nY, 2:4:nF,:) = dLdZ(2:2:nX*2, 1:2:nY*2, 1:fix(nF/4),:);
%                 dLdX1(1:nX, 1:nY, 3:4:nF,:) = dLdZ(1:2:nX*2, 2:2:nY*2, 1:fix(nF/4),:);
%                 dLdX1(1:nX, 1:nY, 4:4:nF,:) = dLdZ(2:2:nX*2, 2:2:nY*2, 1:fix(nF/4),:);
%             end
%         end
    end
end

