% M2 Task 4: Projection of the baseline 1-D CNN.
%
% Projection runs principal component analysis (PCA) on the activations
% flowing into and out of each conv/FC layer, using the training data. Each
% layer's weights are then replaced by smaller matrices that only keep the
% directions that carry most of that variance. Unlike magnitude pruning, the
% number of parameters actually goes down, so the model is smaller in memory.
%
% Phase 1 - Sweep: compress at several learnables-reduction goals with no
%           retraining and see how much accuracy survives.
% Phase 2 - Fine-tune: take the most aggressive goal that keeps baseline
%           accuracy, fine-tune it, and unpack it into plain layers.
%
% Run the whole file from the repository root. Needs the "Deep Learning
% Toolbox Model Compression Library" support package (same as M2_03 pruning).

%% 1. SETUP: load the baseline network and data

% Find the repo root from this file's location (falls back to the current
% folder when the code is run line by line in the terminal).
repoRoot = fileparts(fileparts(mfilename("fullpath")));
if isempty(repoRoot), repoRoot = pwd; end
dataDir      = fullfile(repoRoot, "data");
projectedDir = fullfile(dataDir, "projected"); % projected models are saved here
figureDir    = fullfile(repoRoot, "tasks", "milestone_2", "figures");
if ~isfolder(projectedDir), mkdir(projectedDir); end
if ~isfolder(figureDir), mkdir(figureDir); end

assert(exist("compressNetworkUsingProjection", "file") > 0, ...
    "Projection needs the Deep Learning Toolbox Model Compression Library " + ...
    "support package. Install it from Home > Add-Ons, then restart MATLAB.");

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
baseParams = countLearnables(baseNet);
fprintf("Baseline: val accuracy %.4f, confidence %.4f, %d parameters\n", ...
    baseAcc, baseConf, baseParams);

%% 2. NEURON PCA ON THE TRAINING DATA
% neuronPCA pushes the training set through the network and records the
% principal components of every projectable layer's input and output
% activations. It only needs to run once; every compression level below
% reuses it. Validation and test data are never used here.

% neuronPCA reads plain arrays from a minibatchqueue:
% channels (1) x time (5000) x batch, labelled "CTB".
XTrainArray = permute(single(training.trainData), [3 2 1]);
trainStore  = arrayDatastore(XTrainArray, IterationDimension=3);
mbqTrain = minibatchqueue(trainStore, 1, ...
    MiniBatchSize=32, ...
    MiniBatchFcn=@(X) cat(3, X{:}), ...
    MiniBatchFormat="CTB");

npca = neuronPCA(baseNet, mbqTrain);

%% 3. PHASE 1: PROJECTION SWEEP
% For each goal, ask for that fraction of the projectable layers' learnables
% to be removed. No retraining happens here; we only measure how much the
% model tolerates. Most of the savings come from conv2 and conv3, which hold
% almost all of the 50.7K parameters. conv1 has a single input channel and
% the final FC layer is tiny, so there is little to gain there.

reductionGoals = 0.1:0.1:0.9;
numGoals = numel(reductionGoals);
sweepParams = nan(numGoals, 1);
sweepAcc    = nan(numGoals, 1);
sweepConf   = nan(numGoals, 1);

for i = 1:numGoals
    try
        projectedNet = compressNetworkUsingProjection(baseNet, npca, ...
            LearnablesReductionGoal=reductionGoals(i));
    catch err
        % Very high goals can be impossible to reach; skip them.
        fprintf("Goal %.0f%%: skipped (%s)\n", 100 * reductionGoals(i), err.message);
        continue
    end
    sweepParams(i) = countLearnables(projectedNet);
    [sweepAcc(i), sweepConf(i)] = evaluateQuick(projectedNet, XVal, YVal, classNames);
end

actualReduction = 1 - sweepParams / baseParams;
sweepResults = table(reductionGoals.', sweepParams, actualReduction, sweepAcc, sweepConf, ...
    VariableNames=["ReductionGoal", "NumParams", "ActualReduction", ...
    "ValAccuracy", "ValConfidence"]);
disp(sweepResults);

figure;
plot(100 * actualReduction, sweepAcc, "-o", LineWidth=1.5); hold on;
plot(100 * actualReduction, sweepConf, "-s", LineWidth=1.5); hold off;
xlabel("Parameters removed (%)"); ylabel("Validation score");
legend("Accuracy", "Correct-class confidence", Location="southwest");
title("Projection (no retraining)"); ylim([0 1.05]); grid on;
exportgraphics(gcf, fullfile(figureDir, "M2_04_projection_sweep.png"));

%% 4. PHASE 2: PICK A GOAL AND FINE-TUNE
% Keep the most aggressive goal that still matches baseline accuracy, then
% fine-tune so the network can recover confidence lost to the projection.
% Set chosenGoal by hand (e.g. 0.8) to override the automatic pick.

chosenGoal = [];
if isempty(chosenGoal)
    bestIdx = find(sweepAcc >= baseAcc, 1, "last");
    assert(~isempty(bestIdx), "No projection goal kept baseline accuracy.");
    chosenGoal = reductionGoals(bestIdx);
end
fprintf("Chosen learnables reduction goal: %.0f%%\n", 100 * chosenGoal);

projectedNet = compressNetworkUsingProjection(baseNet, npca, ...
    LearnablesReductionGoal=chosenGoal);

% Saved under the name "net", like baseline_net.mat, so all checkpoints load
% the same way and their file sizes are comparable.
projectedFile = fullfile(projectedDir, "projected_net.mat");
net = projectedNet;
save(projectedFile, "net");

% Same fine-tuning settings as the M2_03 Taylor pruning.
fineTuneOptions = trainingOptions("adam", ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=20, ...
    MiniBatchSize=32, ...
    Shuffle="every-epoch", ...
    Plots="none", ...
    Verbose=false);
fineTunedNet = trainnet(XTrain, YTrain, projectedNet, "crossentropy", fineTuneOptions);

% Replace the ProjectedLayer objects with ordinary conv/FC layers. The math
% is unchanged, but plain layers are what code generation for embedded
% targets expects.
fineTunedNet = unpackProjectedLayers(fineTunedNet);

[acc, conf] = evaluateQuick(fineTunedNet, XVal, YVal, classNames);
fprintf("After fine-tuning: val accuracy %.4f, confidence %.4f\n", acc, conf);

fineTunedFile = fullfile(projectedDir, "projected_finetuned_net.mat");
net = fineTunedNet;
save(fineTunedFile, "net");

%% 5. COMPARE BASELINE VS. PROJECTED VS. PROJECTED + FINE-TUNE
% Every model is saved the same way and timed the same way (one sample per
% prediction, median of 100 runs) so the numbers are directly comparable
% with the M2_02 baseline and M2_03 pruning results.
%
% SizeMB is the saved .mat file (compressed by MATLAB). ParamMemoryMB is
% parameters x 4 bytes (single precision), what the model needs in memory
% on a device. Projection removes parameters, so both should shrink.

models = {baseNet, projectedNet, fineTunedNet};
modelNames = ["Baseline"; sprintf("Projected %.0f%%", 100 * chosenGoal); ...
    "Projected + fine-tune"];
modelFiles = [fullfile(dataDir, "baseline_net.mat"); projectedFile; fineTunedFile];

numModels = numel(models);
accuracy = zeros(numModels, 1); macroF1 = zeros(numModels, 1);
confidence = zeros(numModels, 1); numParams = zeros(numModels, 1);
sizeMB = zeros(numModels, 1); msPerSample = zeros(numModels, 1);

addpath(fullfile(repoRoot, "notebooks")); % for M2_02_evaluateModel
for m = 1:numModels
    results = M2_02_evaluateModel(models{m}, XVal, YVal, classNames, modelFiles(m));
    accuracy(m)  = results.summaryTable.Accuracy;
    macroF1(m)   = mean(results.metricsTable.F1, "omitnan");
    sizeMB(m)    = results.summaryTable.SizeMB;
    numParams(m) = results.summaryTable.NumParams;
    [~, confidence(m)] = evaluateQuick(models{m}, XVal, YVal, classNames);
    msPerSample(m) = timeOneSample(models{m}, XVal(1));

    fprintf("\n%s per-class metrics:\n", modelNames(m));
    disp(results.metricsTable(:, ["Class", "Precision", "Recall", "F1"]));
    disp(results.confusionMatrix);
end

paramMemoryMB = numParams * 4 / 1e6;
comparison = table(modelNames, accuracy, macroF1, confidence, numParams, ...
    sizeMB, paramMemoryMB, msPerSample, ...
    VariableNames=["Model", "ValAccuracy", "MacroF1", "ValConfidence", ...
    "NumParams", "SizeMB", "ParamMemoryMB", "MsPerSample"]);
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

function n = countLearnables(net)
n = sum(cellfun(@numel, net.Learnables.Value));
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
