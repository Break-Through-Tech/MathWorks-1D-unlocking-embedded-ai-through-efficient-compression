function results = M2_05_benchmark_quantized()
% Compare baseline and quantized model size and CPU single-sample latency.
% From the repository root:
%   addpath("scripts"); results = M2_05_benchmark_quantized();

repoRoot = fileparts(fileparts(mfilename("fullpath")));
dataDir = fullfile(repoRoot, "data");
baselineFile = fullfile(dataDir, "models", "baseline", "baseline_net.mat");
quantizedFile = fullfile(dataDir, "models", "quantized", "quantized_net.mat");
baseline = load(baselineFile, "net");
quantized = load(quantizedFile, "qNet");
assert(quantizationDetails(quantized.qNet).IsQuantized, ...
    "The quantized checkpoint does not contain a quantized network.");

validation = load(fullfile(dataDir, "val.mat"), "valData");
scaler = load(fullfile(dataDir, "scaler.mat"), "mu", "sigma");
assert(all(isfinite(scaler.sigma) & scaler.sigma ~= 0), ...
    "Scaler contains invalid or zero standard deviations.");
oneSample = {single(((validation.valData(1, :) - scaler.mu) ...
    ./ scaler.sigma).')};

models = {baseline.net; quantized.qNet};
modelNames = ["Baseline"; "Quantized INT8"];
modelFiles = [string(baselineFile); string(quantizedFile)];
checkpointSizeMB = zeros(2, 1);
parameterMemoryMB = zeros(2, 1);
cpuInferenceMsPerSample = zeros(2, 1);

for i = 1:2
    checkpoint = dir(modelFiles(i));
    checkpointSizeMB(i) = checkpoint.bytes / 1e6;
    networkMetrics = estimateNetworkMetrics(models{i});
    parameterMemoryMB(i) = sum(networkMetrics.("ParameterMemory (MB)"));
    cpuInferenceMsPerSample(i) = timeOneSampleOnCpu(models{i}, oneSample);
end

results = table(modelNames, checkpointSizeMB, parameterMemoryMB, ...
    cpuInferenceMsPerSample, VariableNames=["Model", "CheckpointSizeMB", ...
    "ParameterMemoryMB", "CpuInferenceMsPerSample"]);
disp(results);
fprintf("CPU timing: 200 warmups, median of 100 one-sample predictions; preprocessing excluded.\n");
fprintf("Quantized timing is MATLAB fixed-point simulation, not deployed INT8 speed.\n");
end

function ms = timeOneSampleOnCpu(model, oneSample)
for i = 1:200
    minibatchpredict(model, oneSample, ExecutionEnvironment="cpu");
end
elapsedSeconds = zeros(100, 1);
for i = 1:numel(elapsedSeconds)
    timer = tic;
    minibatchpredict(model, oneSample, ExecutionEnvironment="cpu");
    elapsedSeconds(i) = toc(timer);
end
ms = median(elapsedSeconds) * 1000;
end
