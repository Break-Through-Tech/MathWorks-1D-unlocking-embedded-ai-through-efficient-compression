function metrics = M2_02_evaluate_baseline()
% Evaluate the saved baseline on validation data before compression.
% Run from MATLAB with:
%   addpath("scripts");
%   metrics = M2_02_evaluate_baseline();

repoRoot = fileparts(fileparts(mfilename("fullpath")));
dataDir = fullfile(repoRoot, "data");
modelPath = fullfile(dataDir, "baseline_net.mat");

assert(isfile(modelPath), "Baseline checkpoint not found: %s", modelPath);
assert(isfile(fullfile(dataDir, "standardized.mat")), ...
    "Standardized training data not found in %s.", dataDir);
assert(isfile(fullfile(dataDir, "val.mat")), ...
    "Validation data not found in %s.", dataDir);
assert(isfile(fullfile(dataDir, "scaler.mat")), ...
    "Training scaler not found in %s.", dataDir);

baseline = load(modelPath, "net");
training = load(fullfile(dataDir, "standardized.mat"), ...
    "trainLabels", "trainData");
validation = load(fullfile(dataDir, "val.mat"), "valData", "valLabels");
scaler = load(fullfile(dataDir, "scaler.mat"), "mu", "sigma");

assert(isa(baseline.net, "dlnetwork"), ...
    "Expected baseline_net.mat to contain a dlnetwork named net.");
assert(size(training.trainData, 2) == size(validation.valData, 2), ...
    "Training and validation signal lengths differ.");
assert(numel(scaler.mu) == size(validation.valData, 2) ...
    && numel(scaler.sigma) == size(validation.valData, 2), ...
    "Scaler dimensions do not match validation signal length.");
assert(all(~isnan(scaler.sigma) & ~isinf(scaler.sigma) & scaler.sigma ~= 0), ...
    "Scaler contains invalid or zero standard deviations.");

classNames = categories(categorical(training.trainLabels));
YValidation = categorical(validation.valLabels, classNames);
assert(~any(ismissing(YValidation)), ...
    "Validation labels contain a class not found in training labels.");

XValidation = num2cell(single(((validation.valData - scaler.mu) ...
    ./ scaler.sigma).'), 1).';
YValidation = YValidation(:);

scores = minibatchpredict(baseline.net, XValidation);
YPrediction = scores2label(scores, classNames);
YPrediction = YPrediction(:);

accuracy = mean(YPrediction == YValidation);
confusion = confusionmat(YValidation, YPrediction, ...
    "Order", categorical(classNames));
truePositive = diag(confusion);
predictedCount = sum(confusion, 1).';
actualCount = sum(confusion, 2);

precision = zeros(size(truePositive));
recall = zeros(size(truePositive));
hasPredictions = predictedCount > 0;
hasActualSamples = actualCount > 0;
precision(hasPredictions) = truePositive(hasPredictions) ...
    ./ predictedCount(hasPredictions);
recall(hasActualSamples) = truePositive(hasActualSamples) ...
    ./ actualCount(hasActualSamples);

f1 = zeros(size(truePositive));
hasF1 = precision + recall > 0;
f1(hasF1) = 2 .* precision(hasF1) .* recall(hasF1) ...
    ./ (precision(hasF1) + recall(hasF1));

timingRuns = 100;
warmupCalls = 200;
oneSample = XValidation(1);
% The first ~100 predictions in a MATLAB session are 3-5x slower, so run
% twice that many untimed before measuring.
for warmupIndex = 1:warmupCalls
    minibatchpredict(baseline.net, oneSample);
end
elapsedSeconds = zeros(timingRuns, 1);
for runIndex = 1:timingRuns
    timer = tic;
    minibatchpredict(baseline.net, oneSample);
    elapsedSeconds(runIndex) = toc(timer);
end

modelFile = dir(modelPath);
metrics = struct( ...
    "evaluationSplit", "validation", ...
    "accuracy", accuracy, ...
    "macroF1", mean(f1), ...
    "perClassF1", table(classNames, precision, recall, f1, ...
        VariableNames=["Class", "Precision", "Recall", "F1"]), ...
    "confusionMatrix", confusion, ...
    "modelFileSizeMB", modelFile.bytes / 1e6, ...
    "inferenceMsPerSample", median(elapsedSeconds) * 1000, ...
    "timingRuns", timingRuns);

fprintf("Baseline validation accuracy: %.4f\n", metrics.accuracy);
fprintf("Baseline validation macro F1: %.4f\n", metrics.macroF1);
disp(metrics.perClassF1);
fprintf("Checkpoint file size: %.4f MB\n", metrics.modelFileSizeMB);
fprintf("Inference time: %.4f ms/sample (median of %d runs)\n", ...
    metrics.inferenceMsPerSample, metrics.timingRuns);
fprintf("Timing uses one standardized sample per prediction; preprocessing is excluded.\n");
end
