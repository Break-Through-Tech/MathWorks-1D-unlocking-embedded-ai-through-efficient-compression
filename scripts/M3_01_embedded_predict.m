function scores = M3_01_embedded_predict(rawSignal)
%#codegen
persistent net mu sigma
if isempty(net)
    net = coder.loadDeepLearningNetwork( ...
        "data/models/projected/projected_finetuned_net.mat");
    scaler = coder.load("data/scaler.mat", "mu", "sigma");
    mu = single(scaler.mu);
    sigma = single(scaler.sigma);
end

standardizedSignal = (reshape(single(rawSignal), 1, 5000) - mu) ./ sigma;
formattedSignal = dlarray(standardizedSignal, "CT");
dlScores = predict(net, formattedSignal);
scores = reshape(extractdata(dlScores), 1, 3);
end