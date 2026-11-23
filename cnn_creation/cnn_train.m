
clc;
clear;
close all;

%% ============================================================
% RUMC THREE-CLASS CNN
%
% Classes:
%   1. No_lesion
%   2. Hypo_lesion
%   3. Hyper_lesion
%
% Input:
%   Raw USDATA MAT files
%   Common USHEADER file
%
% Preprocessing:
%   preprocessing()
%   FK Migration
%   Hilbert Envelope
%   Normalization
%   enhancement2()
%
% CNN:
%   Four convolutional blocks
%   Adam optimizer
%
% Output:
%   Accuracy
%   Per-class Precision
%   Per-class Sensitivity
%   Per-class Specificity
%   F1-score
%   Confusion Matrix
%   Trained CNN
%% ============================================================


%% ------------------------------------------------------------
% 1. DATASET CONFIGURATION
% ------------------------------------------------------------

datasetPath = "/MATLAB Drive/dataset";

headerFile = ...
    "/MATLAB Drive/mathwork_competition/creation_of_pipeline/USHEADER_20210624112712.mat";

classNames = ["No_lesion", ...
              "Hypo_lesion", ...
              "Hyper_lesion"];

numClasses = numel(classNames);

%% Load common header

headerData = load(headerFile, "USHEADER");

USHEADER = headerData.USHEADER;


%% ------------------------------------------------------------
% 2. IDENTIFY MAT FILES
% ------------------------------------------------------------

fprintf("\n========== DATASET LOADING ==========\n");

matFiles = cell(0,1);
labels = strings(0,1);

for k = 1:numClasses

    folderPath = fullfile(datasetPath, classNames(k));

    if ~isfolder(folderPath)
        error("Folder not found: %s", folderPath);
    end

    files = dir(fullfile(folderPath, "USDATA*.mat"));

    fprintf("\nClass: %s\n", classNames(k));
    fprintf("Files found: %d\n", numel(files));

    if isempty(files)
        error("No USDATA MAT files found in %s", folderPath);
    end

    for i = 1:numel(files)

        matFiles{end+1,1} = ...
            fullfile(folderPath, files(i).name);

        labels(end+1,1) = classNames(k);

    end

end

totalFiles = numel(matFiles);

Y = categorical(labels, classNames);

fprintf("\nTotal dataset samples = %d\n", totalFiles);

disp("Class distribution:");
disp(countcats(Y));


%% ------------------------------------------------------------
% 3. ULTRASOUND PREPROCESSING
% ------------------------------------------------------------

fprintf("\n========== PREPROCESSING ==========\n");

imageCells = cell(totalFiles,1);

for i = 1:totalFiles

    fprintf("Processing %d / %d : %s\n", ...
        i, totalFiles, matFiles{i});

    enhanced_img = processUltrasound( ...
        matFiles{i}, USHEADER);

    imageCells{i} = enhanced_img;

end

fprintf("\nAll ultrasound files reconstructed.\n");


%% ------------------------------------------------------------
% 4. CONVERT IMAGES INTO CNN DATA ARRAY
% ------------------------------------------------------------

fprintf("\n========== CNN DATA PREPARATION ==========\n");

sampleImage = imageCells{1};

inputSize = [size(sampleImage,1), ...
             size(sampleImage,2), ...
             1];

fprintf("CNN input size = %s\n", mat2str(inputSize));

X = zeros(inputSize(1), ...
          inputSize(2), ...
          1, ...
          totalFiles, ...
          "single");

for i = 1:totalFiles

    img = imageCells{i};

    if ~isequal(size(img), inputSize(1:2))
        error("Image dimensions differ at sample %d.", i);
    end

    X(:,:,1,i) = single(img);

end

clear imageCells;

fprintf("CNN data array created.\n");

fprintf("X dimensions: %s\n", mat2str(size(X)));


%% ------------------------------------------------------------
% 5. STRATIFIED DATASET SPLITTING
% 70% Training
% 15% Validation
% 15% Testing
% ------------------------------------------------------------

fprintf("\n========== DATASET SPLITTING ==========\n");

rng(42);

trainIdx = [];
valIdx = [];
testIdx = [];

classOrder = categorical(classNames, classNames);

for k = 1:numClasses

    idx = find(Y == classOrder(k));

    idx = idx(randperm(numel(idx)));

    n = numel(idx);

    nTrain = round(0.70*n);
    nVal = round(0.15*n);

    % Preserve at least one test sample per class
    nTrain = min(nTrain, n-2);
    nVal = min(nVal, n-nTrain-1);

    trainIdx = [trainIdx; idx(1:nTrain)];

    valIdx = [valIdx; ...
              idx(nTrain+1:nTrain+nVal)];

    testIdx = [testIdx; ...
               idx(nTrain+nVal+1:end)];

end

% Shuffle samples within each split

trainIdx = trainIdx(randperm(numel(trainIdx)));
valIdx = valIdx(randperm(numel(valIdx)));
testIdx = testIdx(randperm(numel(testIdx)));

XTrain = X(:,:,:,trainIdx);
YTrain = Y(trainIdx);

XVal = X(:,:,:,valIdx);
YVal = Y(valIdx);

XTest = X(:,:,:,testIdx);
YTest = Y(testIdx);

fprintf("Training   : %d\n", numel(trainIdx));
fprintf("Validation : %d\n", numel(valIdx));
fprintf("Testing    : %d\n", numel(testIdx));

fprintf("\nTraining distribution:\n");
disp(countcats(YTrain));

fprintf("Validation distribution:\n");
disp(countcats(YVal));

fprintf("Testing distribution:\n");
disp(countcats(YTest));


%% ------------------------------------------------------------
% 6. CNN ARCHITECTURE
% ------------------------------------------------------------

layers = [

    imageInputLayer(inputSize, ...
        "Normalization","none", ...
        "Name","input")


    convolution2dLayer(3,16, ...
        "Padding","same", ...
        "Name","conv1")

    batchNormalizationLayer("Name","bn1")

    reluLayer("Name","relu1")

    maxPooling2dLayer(2, ...
        "Stride",2, ...
        "Name","pool1")


    convolution2dLayer(3,32, ...
        "Padding","same", ...
        "Name","conv2")

    batchNormalizationLayer("Name","bn2")

    reluLayer("Name","relu2")

    maxPooling2dLayer(2, ...
        "Stride",2, ...
        "Name","pool2")


    convolution2dLayer(3,64, ...
        "Padding","same", ...
        "Name","conv3")

    batchNormalizationLayer("Name","bn3")

    reluLayer("Name","relu3")

    maxPooling2dLayer(2, ...
        "Stride",2, ...
        "Name","pool3")


    convolution2dLayer(3,128, ...
        "Padding","same", ...
        "Name","conv4")

    batchNormalizationLayer("Name","bn4")

    reluLayer("Name","relu4")


    globalAveragePooling2dLayer("Name","gap")


    fullyConnectedLayer(64, ...
        "Name","fc1")

    reluLayer("Name","relu5")

    dropoutLayer(0.4, ...
        "Name","dropout")


    % Three-class output

    fullyConnectedLayer(3, ...
        "Name","fc2")

    softmaxLayer("Name","softmax")

    classificationLayer("Name","output")

];


%% ------------------------------------------------------------
% 7. DISPLAY CNN ARCHITECTURE
% ------------------------------------------------------------

figure;

plot(layerGraph(layers));

title("RUMC Three-Class CNN");


%% ------------------------------------------------------------
% 8. TRAINING OPTIONS
% ------------------------------------------------------------

miniBatchSize = 8;

options = trainingOptions("adam", "InitialLearnRate",1e-3, "MaxEpochs",30, "MiniBatchSize",miniBatchSize, "Shuffle","every-epoch", "ValidationData",{XVal,YVal}, "ValidationFrequency", max(1,floor(numel(trainIdx)/miniBatchSize)),"Verbose",true,"Plots","training-progress","ExecutionEnvironment","auto");


%% ------------------------------------------------------------
% 9. TRAIN CNN
% ------------------------------------------------------------

fprintf("\n========== CNN TRAINING ==========\n");

net = trainNetwork( ...
    XTrain, ...
    YTrain, ...
    layers, ...
    options);


%% ------------------------------------------------------------
% 10. TEST CNN
% ------------------------------------------------------------

fprintf("\n========== CNN TESTING ==========\n");

YPred = classify(net, XTest);

accuracy = mean(YPred == YTest);

fprintf("\nAccuracy = %.2f %%\n", accuracy*100);


%% ------------------------------------------------------------
% 11. CONFUSION MATRIX
% ------------------------------------------------------------

figure;

%% Confusion Matrix

figure;

cm = confusionchart(YTest, YPred);

% Set the required class order
sortClasses(cm, classOrder);

cm.Title = 'Three-Class Ultrasound Classification';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

cm.Title = "RUMC Three-Class CNN Confusion Matrix";

cm.RowSummary = "row-normalized";

cm.ColumnSummary = "column-normalized";


%% ------------------------------------------------------------
% 12. MULTICLASS PERFORMANCE METRICS
% ------------------------------------------------------------

C = confusionmat(YTest, YPred, ...
    "Order",classOrder);

% One-versus-rest calculations

TP = diag(C);

FN = sum(C,2) - TP;

FP = sum(C,1)' - TP;

TN = sum(C(:)) - TP - FN - FP;


% Sensitivity / Recall

sensitivity = TP ./ max(TP + FN, eps);


% Specificity

specificity = TN ./ max(TN + FP, eps);


% Precision

precision = TP ./ max(TP + FP, eps);


% F1-score

f1score = 2 .* precision .* sensitivity ./ ...
    max(precision + sensitivity, eps);


%% ------------------------------------------------------------
% 13. DISPLAY RESULTS
% ------------------------------------------------------------

fprintf("\n========== FINAL RESULTS ==========\n");

fprintf("Overall Accuracy = %.2f %%\n", ...
    accuracy*100);

results = table( ...
    classNames(:), ...
    precision*100, ...
    sensitivity*100, ...
    specificity*100, ...
    f1score*100, ...
    'VariableNames', ...
    {'Class','Precision','Sensitivity','Specificity','F1Score'});

disp(results);

fprintf("\nConfusion Matrix:\n");
disp(C);


%% ------------------------------------------------------------
% 14. SAVE TRAINED MODEL
% ------------------------------------------------------------

modelFile = fullfile(datasetPath, ...
    "RUMC_ThreeClass_CNN.mat");

save(modelFile, ...
    "net", ...
    "accuracy", ...
    "precision", ...
    "sensitivity", ...
    "specificity", ...
    "f1score", ...
    "inputSize", ...
    "classNames", ...
    "C", ...
    "trainIdx", ...
    "valIdx", ...
    "testIdx");

fprintf("\nModel saved to:\n%s\n", modelFile);

fprintf("\n========== PIPELINE COMPLETE ==========\n");


%% ============================================================
% LOCAL PREPROCESSING FUNCTION
% ============================================================


function enhanced_img = processUltrasound(matFile, USHEADER)

    %% Load raw signal

    data = load(matFile, "USDATA");

    if ~isfield(data, "USDATA")
        error("USDATA not found in %s", matFile);
    end

    sig = squeeze(data.USDATA);

    %% Reconstruction parameters

    param.x = linspace(-0.0125, 0.0125, 128);
    param.y = linspace(0, 0.0725, 500);

    param.fs = 31250000;
    param.c = 1540;
    param.p = 0;

    param.TXangle = deg2rad(USHEADER.xmitAngles);
    param.pitch = 1.953000000000000e-04;

    param.passband = [0.3, 0.7];
    param.ncol = 128;
    param.to = 0;

    %% Verify acquisition compatibility

    if size(sig,3) ~= numel(param.TXangle)
        error("Signal/header angle mismatch in %s", matFile);
    end

    %% Preprocessing

    [sig, param] = preprocessing( ...
        sig, USHEADER, param);

    %% FK Migration

    [migSIG, param] = fkmig(sig, param);

    %% Envelope detection

    env = abs(hilbert(migSIG));

    %% Normalize

    env = env ./ (max(env(:)) + eps);

    %% Uniform +10 dB Gain

    gain_dB = 10;
    gain = 10^(gain_dB / 20);

    env_enhanced = env * gain;

    %% Log compression

    env_dB = 20 * log10(env_enhanced + eps);

    %% Four-stage enhancement

    enhanced_img = enhancement2(env_enhanced);

    %% Validate output

    enhanced_img = squeeze(enhanced_img);

    if isempty(enhanced_img) || ...
       ~ismatrix(enhanced_img) || ...
       any(~isfinite(enhanced_img(:)))

        error("Invalid reconstructed image: %s", matFile);

    end

    enhanced_img = single(enhanced_img);

end
