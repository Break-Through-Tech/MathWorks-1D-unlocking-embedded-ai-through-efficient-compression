function results = evaluateModel(net, XTest, YTest, classNames, modelFilePath)
% EVALUATEMODEL Computes accuracy, per-class precision/recall/TPR/FPR/F1,
% model size, and inference speed for a trained network. Returns a
% struct containing a per-class metrics table and an overall summary table.

% --- Predictions + timing ---
tic;
scores = minibatchpredict(net, XTest);
elapsed = toc;

YPred = scores2label(scores, classNames);

% --- Overall accuracy ---
accuracy = mean(YPred == YTest);

% --- Confusion matrix ---
confMat = confusionmat(YTest, YPred);
numClasses = numel(classNames);
total = sum(confMat(:));

TP = zeros(numClasses, 1);
FP = zeros(numClasses, 1);
FN = zeros(numClasses, 1);
TN = zeros(numClasses, 1);

for i = 1:numClasses
    TP(i) = confMat(i, i);
    FP(i) = sum(confMat(:, i)) - TP(i);
    FN(i) = sum(confMat(i, :)) - TP(i);
    TN(i) = total - TP(i) - FP(i) - FN(i);
end

precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);      % this is TPR
TPR = recall;
FPR = FP ./ (FP + TN);
specificity = TN ./ (TN + FP); % TNR
f1 = 2 * (precision .* recall) ./ (precision + recall);

% --- Per-class metrics table ---
metricsTable = table(classNames, TP, FP, FN, TN, precision, recall, ...
    TPR, FPR, specificity, f1, ...
    'VariableNames', {'Class','TP','FP','FN','TN','Precision','Recall', ...
    'TPR','FPR','Specificity','F1'});

% --- Model size ---
d = dir(modelFilePath);
sizeMB = d.bytes / 1e6;

% --- Parameter count ---
numParams = sum(cellfun(@numel, net.Learnables.Value));

% --- Inference speed ---
msPerSample = (elapsed / numel(YTest)) * 1000;

% --- Overall summary table (one row) ---
summaryTable = table(accuracy, sizeMB, numParams, msPerSample, ...
    'VariableNames', {'Accuracy','SizeMB','NumParams','MsPerSample'});

% --- Package results ---
results.summaryTable = summaryTable;
results.metricsTable = metricsTable;
results.confusionMatrix = confMat;
end
