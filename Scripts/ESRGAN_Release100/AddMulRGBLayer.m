classdef AddMulRGBLayer < nnet.layer.Layer

    properties
        AddRGB
        MulRGB
    end

    properties (Learnable)
    end
    
    methods
        function layer = AddMulRGBLayer(name, aAddRGB, aMulRGB)
            layer.NumInputs = 1;
            layer.Name = name;
            layer.AddRGB = aAddRGB;
            layer.MulRGB = aMulRGB;
            layer.Description = "Add " + num2str(aAddRGB) + " then mul " + num2str(aMulRGB);
        end
        
        function Z = forward(layer, X1)
            Z = predict(layer,X1);
        end
        
        function Z = predict(layer, X1)
            A = layer.AddRGB;
            M = layer.MulRGB;            

%             if A < 0
%                 meanV = mean(X1, 'all');
%                 stdV = std(X1, 1, 'all');
%                 fprintf('X1, mean, %f, std, %f\n', meanV, stdV);
%             end
            
            % S S C B
            Z = X1;
            Z(:,:,1,:) = (X1(:,:,1,:) + A(1)) * M(1);
            Z(:,:,2,:) = (X1(:,:,2,:) + A(2)) * M(2);
            Z(:,:,3,:) = (X1(:,:,3,:) + A(3)) * M(3);
            
%              if A < 0
%                  meanV = mean(Z, 'all');
%                  stdV = std(Z, 1, 'all');
%                  fprintf('Z, mean, %f, std, %f\n', meanV, stdV);
%              end
        end
    end
end

