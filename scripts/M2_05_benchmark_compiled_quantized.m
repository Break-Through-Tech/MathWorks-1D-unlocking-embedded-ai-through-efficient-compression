function results = M2_05_benchmark_compiled_quantized()
% Generate CPU MEX functions and compare baseline and INT8 inference speed.
% Requires MATLAB R2026b+, MATLAB Coder, its Deep Learning interface,
% Deep Learning Toolbox, Fixed-Point Designer, and Model Compression Library.
% From the project root:
%   addpath("scripts"); results = M2_05_benchmark_compiled_quantized();

repoRoot = fileparts(fileparts(mfilename("fullpath")));
previousDir = pwd;
restoreDir = onCleanup(@() cd(previousDir));
cd(repoRoot);
addpath(fullfile(repoRoot, "scripts"));

assert(license("test", "MATLAB_Coder") == 1, ...
    "MATLAB Coder is required to compile the models.");
assert(exist("analyzeNetworkForCodegen", "file") > 0, ...
    "Quantized C code generation requires MATLAB R2026b or newer.");

dataDir = fullfile(repoRoot, "data");
baselineFile = fullfile(dataDir, "models", "baseline", "baseline_net.mat");
quantizedFile = fullfile(dataDir, "models", "quantized", "quantized_net.mat");
if ~isfile(baselineFile)
    source = fullfile(dataDir, "baseline_net.mat");
    assert(isfile(source), "Missing baseline checkpoint: %s", source);
    mkdir(fileparts(baselineFile));
    copyfile(source, baselineFile);
end
if ~isfile(quantizedFile)
    source = fullfile(dataDir, "quantized_net.mat");
    assert(isfile(source), "Missing quantized checkpoint: %s", source);
    mkdir(fileparts(quantizedFile));
    copyfile(source, quantizedFile);
end

baseline = load(baselineFile, "net");
quantized = load(quantizedFile, "qNet");
assert(quantizationDetails(quantized.qNet).IsQuantized, ...
    "The quantized checkpoint does not contain a quantized network.");
baselineSupport = analyzeNetworkForCodegen(baseline.net, TargetLibrary="none");
quantizedSupport = analyzeNetworkForCodegen(quantized.qNet, TargetLibrary="none");
disp(baselineSupport);
disp(quantizedSupport);
assert(all(baselineSupport.Supported) ...
    && all(quantizedSupport.Supported), ...
    "One of the networks is not supported for generic C code generation.");

config = coder.config("mex");
config.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary="none");
fprintf("Compiling baseline CPU MEX...\n");
codegen -config config M2_05_baseline_predict ...
    -args {zeros(1, 5000, "single")}
fprintf("Compiling quantized CPU MEX...\n");
codegen -config config M2_05_quantized_predict ...
    -args {zeros(1, 5000, "single")}

validation = load(fullfile(dataDir, "val.mat"), "valData", "valLabels");
scaler = load(fullfile(dataDir, "scaler.mat"), "mu", "sigma");
training = load(fullfile(dataDir, "standardized.mat"), "trainLabels");
assert(all(isfinite(scaler.sigma) & scaler.sigma ~= 0), ...
    "Scaler contains invalid or zero standard deviations.");
signals = single((validation.valData - scaler.mu) ./ scaler.sigma);
classNames = categories(categorical(training.trainLabels));
labels = categorical(validation.valLabels, classNames);
assert(~any(ismissing(labels)), "Validation contains an unknown class.");

numSamples = size(signals, 1);
baselineScores = zeros(numSamples, numel(classNames), "single");
quantizedScores = zeros(numSamples, numel(classNames), "single");
for i = 1:numSamples
    baselineScores(i, :) = M2_05_baseline_predict_mex(signals(i, :));
    quantizedScores(i, :) = M2_05_quantized_predict_mex(signals(i, :));
end
baselinePred = scores2label(baselineScores, classNames);
quantizedPred = scores2label(quantizedScores, classNames);
matlabSignals = num2cell(signals.', 1).';
matlabScores = minibatchpredict(quantized.qNet, matlabSignals, ...
    ExecutionEnvironment="cpu");
matlabPred = scores2label(matlabScores, classNames);
assert(isequal(quantizedPred(:), matlabPred(:)), ...
    "Compiled INT8 predictions disagree with the MATLAB quantized network.");

modelNames = ["Baseline MEX"; "Quantized INT8 MEX"];
baselineCheckpoint = dir(baselineFile);
quantizedCheckpoint = dir(quantizedFile);
checkpointSizeMB = [baselineCheckpoint.bytes; quantizedCheckpoint.bytes] / 1e6;
baselineMetrics = estimateNetworkMetrics(baseline.net);
quantizedMetrics = estimateNetworkMetrics(quantized.qNet);
parameterMemoryMB = [sum(baselineMetrics.("ParameterMemory (MB)")); ...
    sum(quantizedMetrics.("ParameterMemory (MB)"))];
validationAccuracy = [mean(baselinePred(:) == labels(:)); ...
    mean(quantizedPred(:) == labels(:))];
oneSample = signals(1, :);
cpuMsPerSample = [timeOneSample(@M2_05_baseline_predict_mex, oneSample); ...
    timeOneSample(@M2_05_quantized_predict_mex, oneSample)];
results = table(modelNames, checkpointSizeMB, parameterMemoryMB, ...
    validationAccuracy, cpuMsPerSample, ...
    VariableNames=["Model", "CheckpointSizeMB", "ParameterMemoryMB", ...
    "ValidationAccuracy", "CpuMsPerSample"]);
disp(results);
fprintf("Compiled CPU timing: 20 warmups, median of 30 one-sample calls; preprocessing and compilation excluded.\n");
fprintf("Baseline/INT8 speed ratio: %.2fx (above 1 means INT8 is faster).\n", ...
    cpuMsPerSample(1) / cpuMsPerSample(2));
end

function ms = timeOneSample(predictFunction, oneSample)
for i = 1:20
    predictFunction(oneSample);
end
elapsedSeconds = zeros(30, 1);
for i = 1:numel(elapsedSeconds)
    timer = tic;
    predictFunction(oneSample);
    elapsedSeconds(i) = toc(timer);
end
ms = median(elapsedSeconds) * 1000;
end
