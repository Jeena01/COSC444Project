classdef NopLayer < nnet.layer.Layer

    properties
    end

    properties (Learnable)
    end
    
    methods
        function layer = NopLayer(name)
            layer.NumInputs = 1;
            layer.Name = name;
            layer.Description = "Nop";
        end
        
        function Z = predict(layer, X1)
            Z = X1; 
        end
        
        function Z = forward(layer, X1)
            Z = predict(layer, X1);
        end
    end
end