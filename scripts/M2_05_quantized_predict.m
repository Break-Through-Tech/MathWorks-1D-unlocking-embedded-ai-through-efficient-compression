function scores = M2_05_quantized_predict(standardizedSignal)
%#codegen
% Inference entry point for the quantized model on one standardized signal.
persistent qNet
if isempty(qNet)
    qNet = coder.loadDeepLearningNetwork( ...
        "data/models/quantized/quantized_net.mat");
end

formattedSignal = dlarray(reshape(single(standardizedSignal), 1, 5000), "CT");
dlScores = predict(qNet, formattedSignal);
scores = reshape(extractdata(dlScores), 1, 3);
end
