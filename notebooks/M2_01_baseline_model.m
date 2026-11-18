% load data
raw     = load("train.mat");
rawVal  = load("val.mat");
rawTest = load("test.mat");

% global standardize
% one mean and one std for the whole training set!!! (instead of per col)
mu    = mean(raw.trainData, "all");
sigma = std(raw.trainData, 1, "all");     

trainData  = (raw.trainData    - mu) / sigma;
valDataStd = (rawVal.valData   - mu) / sigma;
testData   = (rawTest.testData - mu) / sigma;

% labels
YTrain = categorical(raw.trainLabels);
YTest  = categorical(rawTest.testLabels);
YVal   = categorical(rawVal.valLabels, categories(YTrain));
summary(YTrain), summary(YVal), summary(YTest)  

% sanity check...should print about 0 and 1
% mean(trainData(:)), std(trainData(:), 1)

save("scaler.mat", "mu", "sigma");        % now two single numbers

% 3. reshape data, was getting errors beforehand
XTrain = num2cell(trainData.', 1).';   
XVal   = num2cell(valDataStd.', 1).';
XTest  = num2cell(testData.', 1).';

classNames = categories(YTrain);

size(XTrain{1})
size(XTrain)

% define the 1d cnn (baseline)
numClasses = numel(classNames);

layers = [
    sequenceInputLayer(1, MinLength=5000)

    convolution1dLayer(64, 16, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(4, Stride=4)

    convolution1dLayer(32, 32, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(4, Stride=4)

    convolution1dLayer(16, 64, Padding="same")
    batchNormalizationLayer
    reluLayer

    globalAveragePooling1dLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(numClasses)
    softmaxLayer];

%analyzeNetwork(layers)
%%
% train
options = trainingOptions("adam", ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=30, ...
    MiniBatchSize=32, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal, YVal}, ...
    ValidationFrequency=13, ...
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false);

net = trainnet(XTrain, YTrain, layers, "crossentropy", options);
%%
% way to test
scores  = minibatchpredict(net, XTest);
YPred   = scores2label(scores, classNames);
testAcc = mean(YPred == YTest)

figure; confusionchart(YTest, YPred);

numLearnables = sum(cellfun(@numel, net.Learnables.Value))

save("baseline_net.mat", "net", "testAcc");
d = dir("baseline_net.mat"); sizeMB = d.bytes / 1e6
