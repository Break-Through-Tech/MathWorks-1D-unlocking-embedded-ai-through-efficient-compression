% Post-training INT8 quantization of the saved M2 baseline 1-D CNN.
% From the repository root: run("scripts/M2_02_quantize_model.m")
% Requires MATLAB R2024b+, Deep Learning Toolbox, the Model Compression
% Library add-on, and Fixed-Point Designer for MATLAB simulation.

repoRoot = fileparts(fileparts(mfilename("fullpath")));
dataDir = fullfile(repoRoot, "data");
modelsDir = fullfile(dataDir, "models"); % all model checkpoints live here
quantizedDir = fullfile(modelsDir, "quantized");
if ~isfolder(quantizedDir), mkdir(quantizedDir); end

baseline = load(fullfile(modelsDir, "baseline", "baseline_net.mat"), "net");
standardized = load(fullfile(dataDir, "standardized.mat"), ...
    "trainData", "trainLabels");
validation = load(fullfile(dataDir, "val.mat"), "valData", "valLabels");
scaler = load(fullfile(dataDir, "scaler.mat"), "mu", "sigma");

assert(isa(baseline.net, "dlnetwork"), "Expected a saved dlnetwork.");
assert(all(scaler.sigma ~= 0), "Scaler has zero standard deviations.");
assert(size(standardized.trainData, 2) == size(validation.valData, 2), ...
    "Training and validation signal lengths differ.");

classNames = categories(categorical(standardized.trainLabels));
XCalibration = num2cell(single(standardized.trainData.'), 1).';
XValidation = num2cell(single(((validation.valData - scaler.mu) ...
    ./ scaler.sigma).'), 1).';
YValidation = categorical(validation.valLabels, classNames);
assert(~any(ismissing(YValidation)), "Validation contains an unknown class.");

% prepareNetwork fuses convolution and batch normalization and removes
% inference-only layers such as dropout before collecting activation ranges.
quantObj = dlquantizer(baseline.net, ExecutionEnvironment="MATLAB");
prepareNetwork(quantObj);
calibrate(quantObj, XCalibration, UseGPU="off");
qNet = quantize(quantObj);
details = quantizationDetails(qNet);
assert(details.IsQuantized, "The output network was not quantized.");
assert(~isempty(details.QuantizedLearnables), ...
    "The output network has no quantized learnables.");

baselineScores = minibatchpredict(baseline.net, XValidation);
quantizedScores = minibatchpredict(qNet, XValidation);
baselinePred = scores2label(baselineScores, classNames);
quantizedPred = scores2label(quantizedScores, classNames);

baselineAccuracy = mean(baselinePred(:) == YValidation(:));
quantizedAccuracy = mean(quantizedPred(:) == YValidation(:));
fprintf("Validation accuracy: baseline %.4f; INT8 %.4f\n", ...
    baselineAccuracy, quantizedAccuracy);

% Compare model parameter memory, not MAT-file size, to measure the
% reduction from quantization. A saved dlnetwork has serialization overhead.
[baselineMetrics, quantizedMetrics] = estimateNetworkMetrics(baseline.net, qNet);
baselineParameterMB = sum(baselineMetrics.("ParameterMemory (MB)"));
quantizedParameterMB = sum(quantizedMetrics.("ParameterMemory (MB)"));
fprintf("Estimated parameter memory: baseline %.4f MB; INT8 %.4f MB\n", ...
    baselineParameterMB, quantizedParameterMB);

outputFile = fullfile(quantizedDir, "quantized_net.mat");
save(outputFile, "qNet", "-v7.3");
saved = load(outputFile, "qNet");
assert(quantizationDetails(saved.qNet).IsQuantized, ...
    "The saved network did not reload as quantized.");
baselineFile = dir(fullfile(modelsDir, "baseline", "baseline_net.mat"));
quantizedFile = dir(outputFile);
fprintf("Checkpoint file size (includes MATLAB overhead): baseline %.1f KiB; INT8 %.1f KiB\n", ...
    baselineFile.bytes / 1024, quantizedFile.bytes / 1024);
fprintf("Saved %s\n", outputFile);
