% Generate C code and verify it against the MATLAB network.
% Run from the repository root in MATLAB or MATLAB Online.

repoRoot = fileparts(fileparts(mfilename("fullpath")));
if isempty(repoRoot), repoRoot = pwd; end
originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(repoRoot);

assert(license("test", "MATLAB_Coder") == 1, ...
    "MATLAB Coder license is unavailable for this account.");
assert(license("test", "Neural_Network_Toolbox") == 1, ...
    "Deep Learning Toolbox license is unavailable for this account.");
assert(exist("codegen", "file") > 0, ...
    "MATLAB Coder's codegen command is unavailable.");

modelFile = fullfile("data", "models", "projected", ...
    "projected_finetuned_net.mat");
scalerFile = fullfile("data", "scaler.mat");
testFile = fullfile("data", "test.mat");
assert(isfile(modelFile), "Missing model checkpoint: %s", modelFile);
assert(isfile(scalerFile), "Missing training scaler: %s", scalerFile);
assert(isfile(testFile), "Missing test data: %s", testFile);

fprintf("MATLAB release: %s\n", version("-release"));
disp(ver("coder"));
disp(ver("deeplearning"));

fprintf("Generating a MEX target first to verify predictions...\n");
addpath(fullfile(repoRoot, "scripts"));
codegen -config:mex M3_01_embedded_predict ...
    -args {zeros(1, 5000, "single")} -report

savedModel = load(modelFile, "net");
scaler = load(scalerFile, "mu", "sigma");
testData = load(testFile, "testData", "testLabels");
net = savedModel.net;
classNames = categories(categorical(testData.testLabels));
YTest = categorical(testData.testLabels, classNames);
assert(~any(ismissing(YTest)), "Test labels contain an unknown class.");
assert(all(isfinite(scaler.sigma) & scaler.sigma ~= 0), ...
    "Scaler contains invalid or zero standard deviations.");

testDataStandardized = (testData.testData - scaler.mu) ./ scaler.sigma;
XTest = num2cell(single(testDataStandardized.'), 1).';
matlabScores = minibatchpredict(net, XTest);
matlabPredictions = categorical(scores2label(matlabScores, classNames), classNames);

generatedScores = zeros(size(matlabScores), "single");
for i = 1:size(testData.testData, 1)
    generatedScores(i, :) = M3_01_embedded_predict_mex( ...
        single(testData.testData(i, :)));
end
generatedPredictions = categorical( ...
    scores2label(generatedScores, classNames), classNames);
maxScoreDifference = max(abs(double(generatedScores(:)) ...
    - double(matlabScores(:))));
scoreTolerance = 1e-4;
assert(maxScoreDifference <= scoreTolerance, ...
    "Generated scores differ from MATLAB by %.7g (tolerance %.7g).", ...
    maxScoreDifference, scoreTolerance);
assert(isequal(generatedPredictions, matlabPredictions), ...
    "Generated and MATLAB predictions disagree.");

matlabAccuracy = mean(matlabPredictions == YTest);
generatedAccuracy = mean(generatedPredictions == YTest);
fprintf("MATLAB test accuracy: %.4f (%d/%d)\n", matlabAccuracy, ...
    sum(matlabPredictions == YTest), numel(YTest));
fprintf("Generated MEX test accuracy: %.4f (%d/%d)\n", generatedAccuracy, ...
    sum(generatedPredictions == YTest), numel(YTest));
fprintf("Maximum absolute score difference: %.7g\n", maxScoreDifference);
fprintf("Generated and MATLAB class predictions match on all %d test samples.\n", ...
    numel(YTest));

fprintf("Generating a standalone library and C source...\n");
codegen -config:lib M3_01_embedded_predict ...
    -args {zeros(1, 5000, "single")} -report
fprintf("Code generation complete. Inspect codegen\\lib\\M3_01_embedded_predict\\.\n");