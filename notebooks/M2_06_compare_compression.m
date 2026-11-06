% Compare baseline, pruned, projected, and quantized models on validation data.
% Run with: run("notebooks/M2_06_compare_compression.m")

repoRoot = fileparts(fileparts(mfilename("fullpath")));
dataDir = fullfile(repoRoot, "data");
modelsDir = fullfile(dataDir, "models"); % all model checkpoints live here
figureDir = fullfile(repoRoot, "tasks", "milestone_2", "figures");
if ~isfolder(figureDir)
    mkdir(figureDir);
end

modelNames = ["Baseline"; "Taylor-pruned"; "Projected + fine-tuned"; "Quantized INT8"];
modelFiles = [
    fullfile(modelsDir, "baseline", "baseline_net.mat")
    fullfile(modelsDir, "pruned", "taylor_pruned_net.mat")
    fullfile(modelsDir, "projected", "projected_finetuned_net.mat")
    fullfile(modelsDir, "quantized", "quantized_net.mat")
];
modelVariables = ["net"; "net"; "net"; "qNet"];

for i = 1:numel(modelFiles)
    assert(isfile(modelFiles(i)), ...
        "Missing %s checkpoint. Run the corresponding Task 3/4/5 script first.", ...
        modelFiles(i));
end

training = load(fullfile(dataDir, "standardized.mat"), "trainLabels");
validation = load(fullfile(dataDir, "val.mat"), "valData", "valLabels");
scaler = load(fullfile(dataDir, "scaler.mat"), "mu", "sigma");
classNames = categories(categorical(training.trainLabels));
YValidation = categorical(validation.valLabels, classNames);
assert(~any(ismissing(YValidation)), ...
    "Validation labels contain a class not found in training labels.");
assert(all(isfinite(scaler.sigma) & scaler.sigma ~= 0), ...
    "Scaler contains invalid or zero standard deviations.");

XValidation = num2cell(single(((validation.valData - scaler.mu) ...
    ./ scaler.sigma).'), 1).';
YValidation = YValidation(:);
oneSample = XValidation(1);
numModels = numel(modelFiles);
accuracy = zeros(numModels, 1);
macroF1 = zeros(numModels, 1);
sizeMB = zeros(numModels, 1);
msPerSample = zeros(numModels, 1);
parameterMemoryMB = zeros(numModels, 1);
f1Values = zeros(numModels, numel(classNames));
baseline = load(modelFiles(1), "net");

for i = 1:numModels
    savedModel = load(modelFiles(i), modelVariables(i));
    model = savedModel.(modelVariables(i));
    scores = minibatchpredict(model, XValidation);
    predictions = scores2label(scores, classNames);
    predictions = categorical(predictions(:), classNames);

    confusion = confusionmat(YValidation, predictions, ...
        "Order", categorical(classNames));
    truePositive = diag(confusion);
    precisionDenominator = sum(confusion, 1).';
    recallDenominator = sum(confusion, 2);
    precision = zeros(size(truePositive));
    recall = zeros(size(truePositive));
    hasPredictions = precisionDenominator > 0;
    hasActualSamples = recallDenominator > 0;
    precision(hasPredictions) = truePositive(hasPredictions) ...
        ./ precisionDenominator(hasPredictions);
    recall(hasActualSamples) = truePositive(hasActualSamples) ...
        ./ recallDenominator(hasActualSamples);
    f1 = zeros(size(truePositive));
    hasF1 = precision + recall > 0;
    f1(hasF1) = 2 .* precision(hasF1) .* recall(hasF1) ...
        ./ (precision(hasF1) + recall(hasF1));

    accuracy(i) = mean(predictions == YValidation);
    f1Values(i, :) = f1.';
    macroF1(i) = mean(f1);
    checkpoint = dir(modelFiles(i));
    sizeMB(i) = checkpoint.bytes / 1e6;
    if i == numModels
        [~, networkMetrics] = estimateNetworkMetrics(baseline.net, model);
        parameterMemoryMB(i) = sum(networkMetrics.("ParameterMemory (MB)"));
    else
        parameterMemoryMB(i) = sum(cellfun(@numel, model.Learnables.Value)) ...
            * 4 / 1e6;
    end
    msPerSample(i) = timeOneSample(model, oneSample);
end

comparison = table(modelNames, accuracy, macroF1, sizeMB, parameterMemoryMB, msPerSample, ...
    VariableNames=["Model", "Accuracy", "MacroF1", "CheckpointSizeMB", "ParameterMemoryMB", ...
    "InferenceMsPerSample"]);
perClassF1 = array2table(f1Values, ...
    VariableNames=matlab.lang.makeValidName(classNames));
perClassF1 = addvars(perClassF1, modelNames, Before=1, ...
    NewVariableNames="Model");

disp(comparison);
disp(perClassF1);
writetable(comparison, fullfile(figureDir, "M2_06_comparison.csv"));
writetable(perClassF1, fullfile(figureDir, "M2_06_per_class_f1.csv"));
save(fullfile(figureDir, "M2_06_comparison.mat"), ...
    "comparison", "perClassF1", "classNames");

figureHandle = figure(Visible="off");
layout = tiledlayout(2, 3, TileSpacing="compact", Padding="compact");
title(layout, "Baseline vs. compressed models");

nexttile;
bar(categorical(modelNames), [accuracy, macroF1]);
ylabel("Score");
ylim([0 1]);
title("Validation accuracy and macro F1");
legend("Accuracy", "Macro F1", Location="southoutside");
grid on;

nexttile;
bar(categorical(modelNames), sizeMB);
ylabel("Checkpoint size (MB)");
title("Saved model size");
grid on;

nexttile;
bar(categorical(modelNames), parameterMemoryMB);
ylabel("Parameter memory (MB)");
title("Estimated parameter memory");
grid on;

nexttile;
bar(categorical(modelNames), msPerSample);
ylabel("Milliseconds per sample");
title("Single-sample inference latency");
grid on;

nexttile;
bar(categorical(modelNames), f1Values);
ylabel("F1 score");
ylim([0 1]);
title("Validation F1 by class");
legend(classNames, Location="southoutside");
grid on;

exportgraphics(figureHandle, ...
    fullfile(figureDir, "M2_06_comparison.png"), Resolution=200);
close(figureHandle);
fprintf("Saved comparison tables and plot to %s\n", figureDir);
fprintf("Inference timing uses 200 warmups and the median of 100 single-sample runs.\n");
fprintf("Checkpoint size includes MATLAB serialization overhead; parameter memory is reported separately.\n");

function ms = timeOneSample(model, oneSample)
for i = 1:200
    minibatchpredict(model, oneSample);
end
elapsed = zeros(100, 1);
for i = 1:numel(elapsed)
    timer = tic;
    minibatchpredict(model, oneSample);
    elapsed(i) = toc(timer);
end
ms = median(elapsed) * 1000;
end
