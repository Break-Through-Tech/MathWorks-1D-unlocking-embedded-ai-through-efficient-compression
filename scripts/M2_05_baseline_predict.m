function scores = M2_05_baseline_predict(standardizedSignal)
%#codegen
% Inference entry point for the baseline model on one standardized signal.
persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork( ...
        "data/models/baseline/baseline_net.mat");
end

formattedSignal = dlarray(reshape(single(standardizedSignal), 1, 5000), "CT");
dlScores = predict(net, formattedSignal);
scores = reshape(extractdata(dlScores), 1, 3);
end
