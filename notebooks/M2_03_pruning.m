% M2 Task 3: Pruning the baseline 1-D CNN.
%
% Phase 1 - Magnitude pruning: zero out the smallest weights and see how much
%           accuracy survives. Cheap, no retraining, but does NOT shrink the
%           model in MATLAB (the zeros are still stored and multiplied).
% Phase 2 - Taylor pruning: remove whole conv filters that matter least to the
%           loss, then fine-tune. The network itself gets smaller, so size and
%           speed actually improve.
%
% Run the whole file from the repository root. Phase 2 needs the
% "Deep Learning Toolbox Model Compression Library" support package.

%% 1. SETUP: load the baseline network and data

% Find the repo root from this file's location (falls back to the current
% folder when the code is run line by line in the terminal).
repoRoot = fileparts(fileparts(mfilename("fullpath")));
if isempty(repoRoot), repoRoot = pwd; end
dataDir   = fullfile(repoRoot, "data");
prunedDir = fullfile(dataDir, "pruned"); % pruned models are saved here
figureDir = fullfile(repoRoot, "tasks", "milestone_2", "figures");
if ~isfolder(prunedDir), mkdir(prunedDir); end
if ~isfolder(figureDir), mkdir(figureDir); end

rng(0); % same shuffling and fine-tuning results every run

baseline = load(fullfile(dataDir, "baseline_net.mat"), "net");
baseNet  = baseline.net;

% Training data is already standardized. Validation data is standardized with
% the scaler fitted on training data, exactly like the baseline evaluation.
training   = load(fullfile(dataDir, "standardized.mat"), "trainData", "trainLabels");
validation = load(fullfile(dataDir, "val.mat"), "valData", "valLabels");
scaler     = load(fullfile(dataDir, "scaler.mat"), "mu", "sigma");

classNames = categories(categorical(training.trainLabels));
YTrain = categorical(training.trainLabels, classNames);
YVal   = categorical(validation.valLabels, classNames);

% trainnet/minibatchpredict want one 5000x1 sequence per cell.
valDataStd = (validation.valData - scaler.mu) ./ scaler.sigma;
XTrain = num2cell(training.trainData.', 1).';
XVal   = num2cell(single(valDataStd.'), 1).';

[baseAcc, baseConf] = evaluateQuick(baseNet, XVal, YVal, classNames);
fprintf("Baseline: val accuracy %.4f, correct-class confidence %.4f\n", baseAcc, baseConf);

%% 2. PHASE 1: MAGNITUDE PRUNING SWEEP
% For each sparsity level, zero that fraction of the smallest weights in
% every conv and fully connected layer (per layer, so the tiny first conv and
% final FC layers are never wiped out). Biases and batchnorm are left alone.
% No retraining happens here; we only measure how much the model tolerates.

sparsityLevels = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99];
numLevels = numel(sparsityLevels);
magAcc  = zeros(numLevels, 1);
magConf = zeros(numLevels, 1);
magActualSparsity = zeros(numLevels, 1);

for i = 1:numLevels
    prunedNet = magnitudePrune(baseNet, sparsityLevels(i));
    [magAcc(i), magConf(i)] = evaluateQuick(prunedNet, XVal, YVal, classNames);
    magActualSparsity(i) = weightSparsity(prunedNet);
end

magnitudeResults = table(sparsityLevels.', magActualSparsity, magAcc, magConf, ...
    VariableNames=["TargetSparsity", "ActualSparsity", "ValAccuracy", "ValConfidence"]);
disp(magnitudeResults);

% Keep the most aggressive level that still matches baseline accuracy, so it
% can be compared against Taylor pruning at the end.
bestIdx = find(magAcc >= baseAcc, 1, "last");
bestSparsity = sparsityLevels(bestIdx);
magnitudeNet = magnitudePrune(baseNet, bestSparsity);
fprintf("Highest sparsity keeping baseline accuracy: %.0f%%\n", 100 * bestSparsity);

% Saved under the name "net", like baseline_net.mat, so all checkpoints load
% the same way and their file sizes are comparable.
magnitudeFile = fullfile(prunedDir, "magnitude_pruned_net.mat");
net = magnitudeNet;
save(magnitudeFile, "net");

figure;
plot(100 * sparsityLevels, magAcc, "-o", LineWidth=1.5); hold on;
plot(100 * sparsityLevels, magConf, "-s", LineWidth=1.5); hold off;
xlabel("Weights set to zero (%)"); ylabel("Validation score");
legend("Accuracy", "Correct-class confidence", Location="southwest");
title("Magnitude pruning (no retraining)"); ylim([0 1.05]); grid on;
exportgraphics(gcf, fullfile(figureDir, "M2_03_magnitude_sweep.png"));

%% 3. PHASE 2: TAYLOR (FILTER) PRUNING
% Each pruning iteration:
%   a) trains on a pass of mini-batches while collecting a Taylor score for
%      every conv filter (roughly: how much the loss would change if that
%      filter were removed),
%   b) removes the lowest-scoring filters.
% The training in (a) also lets the network recover from the previous cut.

assert(exist("taylorPrunableNetwork", "file") > 0, ...
    "Taylor pruning needs the Deep Learning Toolbox Model Compression Library " + ...
    "support package. Install it from Home > Add-Ons, then restart MATLAB.");

% 7 iterations x 8 filters = 56 of the 112 conv filters removed (the tool
% treats 91 of them as removable). 10 iterations (80 filters) made the model
% 88% smaller but left confidence at ~0.93 even after fine-tuning; 7 keeps
% it at ~0.99, essentially the baseline's.
maxPruningIterations = 7;
maxToPrune     = 8;        % filters removed per iteration
miniBatchSize  = 32;
learnRate      = 1e-3;
momentum       = 0.9;

prunableNet = taylorPrunableNetwork(baseNet);
numFiltersStart = countFilters(baseNet);

% The custom training loop works on plain arrays, not cells:
% channels (1) x time (5000) x batch, labelled "CTB".
XTrainArray = permute(single(training.trainData), [3 2 1]);
TTrain = onehotencode(YTrain, 2).'; % classes x samples
numTrain = size(XTrainArray, 3);

velocity = [];
taylorHistory = table('Size', [0 4], ...
    'VariableTypes', ["double", "double", "double", "double"], ...
    'VariableNames', ["FiltersLeft", "NumLearnables", "ValAccuracy", "ValConfidence"]);

for iteration = 1:maxPruningIterations
    % a) One shuffled pass over the training set, accumulating filter scores.
    shuffled = randperm(numTrain);
    for first = 1:miniBatchSize:numTrain
        batch = shuffled(first:min(first + miniBatchSize - 1, numTrain));
        X = dlarray(XTrainArray(:, :, batch), "CTB");
        T = dlarray(TTrain(:, batch), "CB");

        [~, pruningGradients, pruningActivations, netGradients, state] = ...
            dlfeval(@modelLossPruning, prunableNet, X, T);
        prunableNet.State = state;
        [prunableNet, velocity] = sgdmupdate(prunableNet, netGradients, ...
            velocity, learnRate, momentum);
        prunableNet = updateScore(prunableNet, pruningActivations, pruningGradients);
    end

    % b) Remove the lowest-scoring filters.
    prunableNet = updatePrunables(prunableNet, MaxToPrune=maxToPrune);

    % The optimizer's momentum no longer matches the smaller layers.
    velocity = [];

    currentNet = dlnetwork(prunableNet);
    [acc, conf] = evaluateQuick(currentNet, XVal, YVal, classNames);
    filtersLeft = countFilters(currentNet);
    taylorHistory(end + 1, :) = {filtersLeft, ...
        countLearnables(currentNet), acc, conf};
    fprintf("Iteration %2d: %3d filters left, val accuracy %.4f, confidence %.4f\n", ...
        iteration, filtersLeft, acc, conf);
end

% Final fine-tune of the pruned network so it can recover any lost accuracy.
% Uses the baseline's learning rate (1e-3): a gentler 1e-4 for 10 epochs only
% brought confidence back to ~0.80 after removing 80 filters.
taylorNet = dlnetwork(prunableNet);
fineTuneOptions = trainingOptions("adam", ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=20, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="none", ...
    Verbose=false);
taylorNet = trainnet(XTrain, YTrain, taylorNet, "crossentropy", fineTuneOptions);

[acc, conf] = evaluateQuick(taylorNet, XVal, YVal, classNames);
fprintf("After fine-tuning: val accuracy %.4f, confidence %.4f\n", acc, conf);
disp(taylorHistory);

taylorFile = fullfile(prunedDir, "taylor_pruned_net.mat");
net = taylorNet;
save(taylorFile, "net");

figure;
filtersRemoved = numFiltersStart - taylorHistory.FiltersLeft;
plot(filtersRemoved, taylorHistory.ValAccuracy, "-o", LineWidth=1.5); hold on;
plot(filtersRemoved, taylorHistory.ValConfidence, "-s", LineWidth=1.5); hold off;
xlabel(sprintf("Conv filters removed (of %d)", numFiltersStart));
ylabel("Validation score");
legend("Accuracy", "Correct-class confidence", Location="southwest");
title("Taylor pruning (before final fine-tune)"); ylim([0 1.05]); grid on;
exportgraphics(gcf, fullfile(figureDir, "M2_03_taylor_pruning.png"));

%% 4. COMPARE BASELINE VS. MAGNITUDE VS. TAYLOR
% Every model is saved the same way and timed the same way (one sample per
% prediction, median of 100 runs) so the numbers are directly comparable
% with the M2_02 baseline results.
%
% Two size columns, because they tell different stories:
%   SizeMB        - the saved .mat file. MATLAB compresses it, so the zeros
%                   from magnitude pruning make the FILE smaller.
%   ParamMemoryMB - parameters x 4 bytes (single precision): what the model
%                   needs in memory on a device. Zeros still take 4 bytes
%                   each, so only removing filters (Taylor) shrinks this.

models = {baseNet, magnitudeNet, taylorNet};
modelNames = ["Baseline"; sprintf("Magnitude %.0f%%", 100 * bestSparsity); "Taylor + fine-tune"];
modelFiles = [fullfile(dataDir, "baseline_net.mat"); magnitudeFile; taylorFile];

numModels = numel(models);
accuracy = zeros(numModels, 1); macroF1 = zeros(numModels, 1);
confidence = zeros(numModels, 1); numParams = zeros(numModels, 1);
nonzeroParams = zeros(numModels, 1); sizeMB = zeros(numModels, 1);
msPerSample = zeros(numModels, 1);

addpath(fullfile(repoRoot, "notebooks")); % for M2_02_evaluateModel
for m = 1:numModels
    results = M2_02_evaluateModel(models{m}, XVal, YVal, classNames, modelFiles(m));
    accuracy(m)  = results.summaryTable.Accuracy;
    macroF1(m)   = mean(results.metricsTable.F1, "omitnan");
    sizeMB(m)    = results.summaryTable.SizeMB;
    numParams(m) = results.summaryTable.NumParams;
    nonzeroParams(m) = sum(cellfun(@(v) nnz(extractdata(v)), models{m}.Learnables.Value));
    [~, confidence(m)] = evaluateQuick(models{m}, XVal, YVal, classNames);
    msPerSample(m) = timeOneSample(models{m}, XVal(1));
end

paramMemoryMB = numParams * 4 / 1e6;
comparison = table(modelNames, accuracy, macroF1, confidence, numParams, ...
    nonzeroParams, sizeMB, paramMemoryMB, msPerSample, ...
    VariableNames=["Model", "ValAccuracy", "MacroF1", "ValConfidence", ...
    "NumParams", "NonzeroParams", "SizeMB", "ParamMemoryMB", "MsPerSample"]);
disp(comparison);

%% LOCAL FUNCTIONS

function [accuracy, confidence] = evaluateQuick(net, X, Y, classNames)
% Validation accuracy plus the average probability the model gives the
% correct class. Confidence drops gradually, so it shows damage that
% accuracy hides when accuracy is stuck at 100%.
scores = minibatchpredict(net, X);
predicted = scores2label(scores, classNames);
accuracy = mean(predicted(:) == Y(:));
correctIdx = sub2ind(size(scores), (1:numel(Y)).', double(Y(:)));
confidence = mean(scores(correctIdx));
end

function net = magnitudePrune(net, sparsity)
% Zero the smallest |weight| values in each conv/FC layer separately.
isWeight = find(net.Learnables.Parameter == "Weights");
for i = isWeight.'
    w = net.Learnables.Value{i};
    numToZero = round(sparsity * numel(w));
    if numToZero == 0, continue; end
    [~, order] = sort(abs(extractdata(w(:))));
    w(order(1:numToZero)) = 0;
    net.Learnables.Value{i} = w;
end
end

function sparsity = weightSparsity(net)
% Fraction of conv/FC weights that are exactly zero.
weights = net.Learnables.Value(net.Learnables.Parameter == "Weights");
numZero = sum(cellfun(@(w) sum(extractdata(w) == 0, "all"), weights));
sparsity = numZero / sum(cellfun(@numel, weights));
end

function n = countLearnables(net)
n = sum(cellfun(@numel, net.Learnables.Value));
end

function n = countFilters(net)
% Total conv filters. Conv weights are [filterSize x inputChannels x
% numFilters]; FC weights are 2-D, so they are skipped.
weights = net.Learnables.Value(net.Learnables.Parameter == "Weights");
isConv = cellfun(@(w) ndims(w) == 3, weights);
n = sum(cellfun(@(w) size(w, 3), weights(isConv)));
end

function [loss, pruningGradients, pruningActivations, netGradients, state] = ...
    modelLossPruning(prunableNet, X, T)
% Forward pass that also returns the filter activations Taylor pruning
% needs, then gradients of the loss w.r.t. those activations (for scoring)
% and w.r.t. the learnables (for the normal training update).
[Y, state, pruningActivations] = forward(prunableNet, X);
loss = crossentropy(Y, T);
[pruningGradients, netGradients] = dlgradient(loss, pruningActivations, ...
    prunableNet.Learnables);
end

function ms = timeOneSample(net, oneSample)
% Same timing method as M2_02_evaluate_baseline: 200 untimed warm-up
% predictions (the first ~100 in a session are much slower), then the
% median of 100 single-sample predictions.
timingRuns = 100;
warmupCalls = 200;
for warmupIndex = 1:warmupCalls
    minibatchpredict(net, oneSample);
end
elapsed = zeros(timingRuns, 1);
for runIndex = 1:timingRuns
    timer = tic;
    minibatchpredict(net, oneSample);
    elapsed(runIndex) = toc(timer);
end
ms = median(elapsed) * 1000;
end
