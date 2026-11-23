
clc;
clear;
close all;

%% ============================================================
% FINAL RUMC ULTRASOUND IMPLEMENTATION
%
% Pipeline:
% Raw RF Data
%     -> Preprocessing
%     -> FK Migration
%     -> Hilbert Envelope
%     -> Normalization
%     -> +10 dB Gain
%     -> Enhancement
%     -> Three-Class CNN Classification
%     -> B-mode Display
%
% CNN Classes:
%   1. No_lesion
%   2. Hypo_lesion
%   3. Hyper_lesion
%% ============================================================


%% ------------------------------------------------------------
% 1. FILE CONFIGURATION
%% ------------------------------------------------------------

dataFile = ...
    "/MATLAB Drive/dataset/No_lesion/USDATA_20210624113330706.mat";

headerFile = ...
    "/MATLAB Drive/mathwork_competition/creation_of_pipeline/USHEADER_20210624112712.mat";

cnnFile = ...
    "/MATLAB Drive/mathwork_competition/cnn_model_and_details/RUMC_ThreeClass_CNN.mat";


%% ------------------------------------------------------------
% 2. CHECK REQUIRED FILES AND FUNCTIONS
%% ------------------------------------------------------------

if ~isfile(dataFile)
    error("Ultrasound data file not found: %s", dataFile);
end

if ~isfile(headerFile)
    error("Header file not found: %s", headerFile);
end

if ~isfile(cnnFile)
    error("CNN model not found: %s", cnnFile);
end

requiredFunctions = ["preprocessing", "fkmig", "enhancement2"];

for k = 1:numel(requiredFunctions)

    if exist(requiredFunctions(k), "file") ~= 2
        error("Required function not found: %s.m", ...
            requiredFunctions(k));
    end

end


%% ------------------------------------------------------------
% 3. LOAD ULTRASOUND HEADER
%% ------------------------------------------------------------

fprintf("\n========== HEADER LOADING ==========\n");

headerData = load(headerFile, "USHEADER");

if ~isfield(headerData, "USHEADER")
    error("USHEADER variable not found in header file.");
end

USHEADER = headerData.USHEADER;

fprintf("Ultrasound header loaded successfully.\n");


%% ------------------------------------------------------------
% 4. RECONSTRUCTION PARAMETERS
%% ------------------------------------------------------------

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


%% ------------------------------------------------------------
% 5. LOAD TRAINED THREE-CLASS CNN
%% ------------------------------------------------------------

fprintf("\n========== CNN MODEL LOADING ==========\n");

modelData = load(cnnFile, "net");

if ~isfield(modelData, "net")
    error("Variable 'net' not found in CNN MAT file.");
end

net = modelData.net;

inputSize = net.Layers(1).InputSize;

fprintf("CNN loaded successfully.\n");

fprintf("Expected CNN input size: %s\n", ...
    mat2str(inputSize));


%% ------------------------------------------------------------
% 6. LOAD RAW ULTRASOUND DATA
%% ------------------------------------------------------------

fprintf("\n========== ULTRASOUND DATA LOADING ==========\n");

data = load(dataFile, "USDATA");

if ~isfield(data, "USDATA")
    error("USDATA variable not found in data file.");
end

sig = squeeze(data.USDATA);

fprintf("Raw RF data dimensions:\n");
disp(size(sig));


%% ------------------------------------------------------------
% 7. PREPROCESSING
%% ------------------------------------------------------------

fprintf("\n========== PREPROCESSING ==========\n");

% Default: retain all transmission angles.
% Set true to use only zero-degree transmission angle.

useZeroDegreeOnly = false;

if useZeroDegreeOnly

    [sig, param] = preprocessing( ...
        sig, USHEADER, param, 1);

else

    [sig, param] = preprocessing( ...
        sig, USHEADER, param);

end

fprintf("Preprocessing completed.\n");


%% ------------------------------------------------------------
% 8. FK MIGRATION
%% ------------------------------------------------------------

fprintf("\n========== FK MIGRATION ==========\n");

[migSIG, param] = fkmig(sig, param);

if isempty(migSIG) || any(~isfinite(migSIG(:)))
    error("Invalid output from FK migration.");
end

fprintf("FK migration completed.\n");

fprintf("Migrated signal dimensions:\n");
disp(size(migSIG));


%% ------------------------------------------------------------
% 9. HILBERT ENVELOPE DETECTION
%% ------------------------------------------------------------

fprintf("\n========== ENVELOPE DETECTION ==========\n");

env = abs(hilbert(migSIG));

if isempty(env) || any(~isfinite(env(:)))
    error("Invalid envelope output.");
end

fprintf("Envelope detection completed.\n");


%% ------------------------------------------------------------
% 10. ENVELOPE NORMALIZATION
%% ------------------------------------------------------------

env = env ./ (max(env(:)) + eps);


%% ------------------------------------------------------------
% 11. UNIFORM +10 dB GAIN
%% ------------------------------------------------------------

gain_dB = 10;

gain = 10^(gain_dB / 20);

env_enhanced = env * gain;

fprintf("Applied uniform gain: +%.1f dB\n", gain_dB);


%% ------------------------------------------------------------
% 12. LOG COMPRESSION FOR DISPLAY
%% ------------------------------------------------------------

env_dB = 20 * log10(env_enhanced + eps);


%% ------------------------------------------------------------
% 13. DISPLAY FK-MIGRATED IMAGE
%% ------------------------------------------------------------

if numel(param.x) ~= size(env_dB, 2)
    error("Lateral coordinate size does not match migrated image.");
end

if isfield(param, "z") && ...
        numel(param.z) == size(env_dB, 1)

    depthCoordinates = param.z;

elseif isfield(param, "y") && ...
        numel(param.y) == size(env_dB, 1)

    depthCoordinates = param.y;

else

    error("No matching depth coordinate vector for migrated image.");

end

figure("Name", "FK Migrated Ultrasound");

imagesc( ...
    param.x * 1000, ...
    depthCoordinates * 1000, ...
    env_dB);

axis image;
colormap gray;
caxis([-60 0]);

ylim([0 35]);

colorbar;

xlabel("Lateral position (mm)");
ylabel("Depth (mm)");

title("FK Migrated Ultrasound +10 dB Gain");


%% ------------------------------------------------------------
% 14. FOUR-STAGE IMAGE ENHANCEMENT
%% ------------------------------------------------------------

fprintf("\n========== IMAGE ENHANCEMENT ==========\n");

enhanced_img = enhancement2(env_enhanced);

enhanced_img = squeeze(enhanced_img);

if isempty(enhanced_img) || ...
        ~ismatrix(enhanced_img) || ...
        any(~isfinite(enhanced_img(:)))

    error("Invalid output from enhancement2.");

end

enhanced_img = single(enhanced_img);

fprintf("Enhancement completed.\n");

fprintf("Enhanced image dimensions:\n");
disp(size(enhanced_img));


%% ------------------------------------------------------------
% 15. VERIFY ENHANCED IMAGE COORDINATES
%% ------------------------------------------------------------

if numel(param.x) ~= size(enhanced_img, 2)
    error("Enhanced image width does not match lateral coordinates.");
end

if isfield(param, "y") && ...
        numel(param.y) == size(enhanced_img, 1)

    enhancedDepth = param.y;

elseif isfield(param, "z") && ...
        numel(param.z) == size(enhanced_img, 1)

    enhancedDepth = param.z;

else

    error("No matching depth coordinate vector for enhanced image.");

end


%% ------------------------------------------------------------
% 16. DISPLAY ENHANCED B-MODE IMAGE
%% ------------------------------------------------------------

figure("Name", "Enhanced Ultrasound B-mode");

imagesc( ...
    param.x * 1000, ...
    enhancedDepth * 1000, ...
    enhanced_img);

axis image;
colormap gray;
caxis([-60 0]);

ylim([0 35]);

colorbar;

xlabel("Lateral position (mm)");
ylabel("Depth (mm)");

title("Enhanced Ultrasound B-mode");


%% ============================================================
% 17. CNN INPUT PREPARATION
%
% IMPORTANT:
% The CNN was trained using:
%   - Full enhanced image
%   - Single precision
%   - No additional min-max normalization
%   - No uint8 conversion
%   - No depth cropping
%% ============================================================

fprintf("\n========== CNN INPUT PREPARATION ==========\n");

if numel(inputSize) ~= 3
    error("Unexpected CNN input dimensions.");
end

if inputSize(3) ~= 1
    error("CNN expects %d channels, but this pipeline supplies 1.", ...
        inputSize(3));
end

if ~isequal(size(enhanced_img), inputSize(1:2))
    error("CNN input dimensions do not match.\nExpected: %s\nActual: %s", ...
        mat2str(inputSize(1:2)), ...
        mat2str(size(enhanced_img)));
end

cnnInput = reshape(enhanced_img, ...
    [inputSize(1), inputSize(2), 1, 1]);

cnnInput = single(cnnInput);

fprintf("CNN input prepared successfully.\n");

fprintf("CNN input dimensions:\n");
disp(size(cnnInput));


%% ------------------------------------------------------------
% 18. THREE-CLASS CNN CLASSIFICATION
%% ------------------------------------------------------------

fprintf("\n========== CNN CLASSIFICATION ==========\n");

[resultLabel, scores] = classify(net, cnnInput);

fprintf("Classification completed.\n");


%% ------------------------------------------------------------
% 19. DISPLAY CLASSIFICATION RESULTS
%% ------------------------------------------------------------

fprintf("\n");
fprintf("============================================\n");
fprintf("       RUMC ULTRASOUND CLASSIFICATION       \n");
fprintf("============================================\n");

fprintf("\nPredicted Class: %s\n", string(resultLabel));

classNames = net.Layers(end).Classes;

fprintf("\nClass Confidence Scores:\n");

for k = 1:numel(classNames)

    fprintf("  %-15s : %.2f %%\n", ...
        string(classNames(k)), ...
        scores(1,k) * 100);

end

fprintf("\n============================================\n");


%% ------------------------------------------------------------
% 20. DISPLAY PREDICTION RESULT
%% ------------------------------------------------------------

figure("Name", "CNN Classification Result");

bar(scores(1,:) * 100);

xticks(1:numel(classNames));

xticklabels(string(classNames));

xtickangle(20);

ylabel("Confidence (%)");

ylim([0 100]);

title("Three-Class CNN Prediction");

grid on;


%% ------------------------------------------------------------
% 21. FINAL COMPLETION
%% ------------------------------------------------------------

fprintf("\n========== PIPELINE COMPLETE ==========\n");

fprintf("Ultrasound preprocessing: SUCCESS\n");
fprintf("FK migration: SUCCESS\n");
fprintf("Envelope detection: SUCCESS\n");
fprintf("Image enhancement: SUCCESS\n");
fprintf("CNN classification: SUCCESS\n");

fprintf("\nFinal Prediction: %s\n", string(resultLabel));

fprintf("\n============================================\n");
