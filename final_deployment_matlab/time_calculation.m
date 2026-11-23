
clc;
clear;
close all;

%% ============================================================
% RUMC ULTRASOUND PIPELINE
% EXECUTION TIME + IMAGE QUALITY BENCHMARK
% NO IMAGE DISPLAY
%% ============================================================

total_timer = tic;

%% 1. FILE CONFIGURATION

dataFile = "/MATLAB Drive/dataset/No_lesion/USDATA_20210624113330706.mat";

headerFile = "/MATLAB Drive/mathwork_competition/final_deployment_matlab/USHEADER_20210624112712.mat";

cnnFile = "/MATLAB Drive/mathwork_competition/cnn_model_and_details/RUMC_ThreeClass_CNN.mat";

%% 2. HEADER LOADING

t = tic;

headerData = load(headerFile, "USHEADER");
USHEADER = headerData.USHEADER;

time_header = toc(t);

%% 3. RECONSTRUCTION PARAMETERS

t = tic;

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

time_parameters = toc(t);

%% 4. CNN MODEL LOADING

t = tic;

modelData = load(cnnFile, "net");
net = modelData.net;

inputSize = net.Layers(1).InputSize;

time_cnn_loading = toc(t);

%% 5. ULTRASOUND DATA LOADING

t = tic;

data = load(dataFile, "USDATA");
sig = squeeze(data.USDATA);

time_data_loading = toc(t);

%% 6. PREPROCESSING

t = tic;

useZeroDegreeOnly = false;

if useZeroDegreeOnly
    [sig, param] = preprocessing(sig, USHEADER, param, 1);
else
    [sig, param] = preprocessing(sig, USHEADER, param);
end

time_preprocessing = toc(t);

%% 7. FK MIGRATION

t = tic;

[migSIG, param] = fkmig(sig, param);

time_fkmig = toc(t);

%% 8. HILBERT ENVELOPE DETECTION

t = tic;

env = abs(hilbert(migSIG));

time_envelope = toc(t);

%% 9. NORMALIZATION

t = tic;

env = env ./ (max(env(:)) + eps);

time_normalization = toc(t);

%% 10. +10 dB GAIN

t = tic;

gain_dB = 10;
gain = 10^(gain_dB / 20);

env_enhanced = env * gain;

time_gain = toc(t);

%% 11. LOG COMPRESSION

t = tic;

env_dB = 20 * log10(env_enhanced + eps);

time_log = toc(t);

%% 12. FOUR-STAGE ENHANCEMENT

t = tic;

enhanced_img = enhancement2(env_enhanced);
enhanced_img = squeeze(enhanced_img);
enhanced_img = single(enhanced_img);

time_enhancement = toc(t);

%% 13. IMAGE QUALITY EVALUATION

t = tic;

% Use the same ROI coordinates for both images.
% Format: [x_start x_end y_start y_end]
%
% IMPORTANT: These are example coordinates.
% Select actual lesion and homogeneous tissue ROIs.

signal_roi = [45 75 180 230];
background_roi = [5 30 180 230];

% Use linear envelope for FK image.
fk_image = env;

% enhancement2 output is retained as returned.
% Check its intensity representation before interpreting metrics.
enh_image = enhanced_img;

% Extract FK ROIs

signal_fk = fk_image(signal_roi(3):signal_roi(4), ...
                     signal_roi(1):signal_roi(2));

background_fk = fk_image(background_roi(3):background_roi(4), ...
                         background_roi(1):background_roi(2));

% Extract enhanced ROIs

signal_enh = enh_image(signal_roi(3):signal_roi(4), ...
                       signal_roi(1):signal_roi(2));

background_enh = enh_image(background_roi(3):background_roi(4), ...
                           background_roi(1):background_roi(2));

% SNR

snr_fk = 10 * log10( ...
    mean(signal_fk(:).^2) / (var(background_fk(:)) + eps));

snr_enh = 10 * log10( ...
    mean(signal_enh(:).^2) / (var(background_enh(:)) + eps));

% CNR

cnr_fk = abs(mean(signal_fk(:)) - mean(background_fk(:))) / ...
    (sqrt((var(signal_fk(:)) + var(background_fk(:))) / 2) + eps);

cnr_enh = abs(mean(signal_enh(:)) - mean(background_enh(:))) / ...
    (sqrt((var(signal_enh(:)) + var(background_enh(:))) / 2) + eps);

% ENL

enl_fk = mean(background_fk(:))^2 / ...
         (var(background_fk(:)) + eps);

enl_enh = mean(background_enh(:))^2 / ...
          (var(background_enh(:)) + eps);

% Mean intensity

mean_fk = mean(fk_image(:));
mean_enh = mean(enh_image(:));

% Standard deviation

std_fk = std(fk_image(:));
std_enh = std(enh_image(:));

% Dynamic range

dynamic_fk = max(fk_image(:)) - min(fk_image(:));
dynamic_enh = max(enh_image(:)) - min(enh_image(:));

time_quality = toc(t);

%% 14. CNN INPUT PREPARATION

t = tic;

if numel(inputSize) ~= 3 || inputSize(3) ~= 1
    error("Unexpected CNN input dimensions.");
end

if ~isequal(size(enhanced_img), inputSize(1:2))
    error("Enhanced image dimensions do not match CNN input.");
end

cnnInput = reshape(enhanced_img, ...
    [inputSize(1), inputSize(2), 1, 1]);

cnnInput = single(cnnInput);

time_cnn_preparation = toc(t);

%% 15. CNN CLASSIFICATION

t = tic;

[resultLabel, scores] = classify(net, cnnInput);

time_cnn_classification = toc(t);

%% 16. TOTAL EXECUTION TIME

total_time = toc(total_timer);

%% ============================================================
% 17. PRINT EXECUTION TIMES
%% ============================================================

fprintf('\n============================================\n');
fprintf('       MATLAB ONLINE EXECUTION TIMES\n');
fprintf('============================================\n');

fprintf('Header Loading             : %.6f sec\n', time_header);
fprintf('Parameter Setup            : %.6f sec\n', time_parameters);
fprintf('CNN Model Loading          : %.6f sec\n', time_cnn_loading);
fprintf('Ultrasound Data Loading    : %.6f sec\n', time_data_loading);
fprintf('Preprocessing              : %.6f sec\n', time_preprocessing);
fprintf('FK Migration               : %.6f sec\n', time_fkmig);
fprintf('Envelope Detection         : %.6f sec\n', time_envelope);
fprintf('Normalization              : %.6f sec\n', time_normalization);
fprintf('Gain Enhancement           : %.6f sec\n', time_gain);
fprintf('Log Compression            : %.6f sec\n', time_log);
fprintf('Four-Stage Enhancement     : %.6f sec\n', time_enhancement);
fprintf('Image Quality Evaluation   : %.6f sec\n', time_quality);
fprintf('CNN Input Preparation      : %.6f sec\n', time_cnn_preparation);
fprintf('CNN Classification        : %.6f sec\n', time_cnn_classification);

fprintf('--------------------------------------------\n');
fprintf('TOTAL EXECUTION TIME       : %.6f sec\n', total_time);

%% ============================================================
% 18. PRINT IMAGE QUALITY METRICS
%% ============================================================

fprintf('\n============================================\n');
fprintf('          IMAGE QUALITY METRICS\n');
fprintf('============================================\n');

fprintf('\n--- FK MIGRATED IMAGE ---\n');

fprintf('SNR            : %.4f dB\n', snr_fk);
fprintf('CNR            : %.6f\n', cnr_fk);
fprintf('ENL            : %.6f\n', enl_fk);
fprintf('Mean           : %.6f\n', mean_fk);
fprintf('Standard Dev   : %.6f\n', std_fk);
fprintf('Dynamic Range  : %.6f\n', dynamic_fk);

fprintf('\n--- ENHANCED IMAGE ---\n');

fprintf('SNR            : %.4f dB\n', snr_enh);
fprintf('CNR            : %.6f\n', cnr_enh);
fprintf('ENL            : %.6f\n', enl_enh);
fprintf('Mean           : %.6f\n', mean_enh);
fprintf('Standard Dev   : %.6f\n', std_enh);
fprintf('Dynamic Range  : %.6f\n', dynamic_enh);

fprintf('\n============================================\n');
fprintf('          PERFORMANCE COMPARISON\n');
fprintf('============================================\n');

fprintf('SNR Difference : %.4f dB\n', snr_enh - snr_fk);
fprintf('CNR Difference : %.6f\n', cnr_enh - cnr_fk);
fprintf('ENL Difference : %.6f\n', enl_enh - enl_fk);

%% ============================================================
% 19. CNN CLASSIFICATION RESULTS
%% ============================================================

classNames = net.Layers(end).Classes;

fprintf('\n============================================\n');
fprintf('          CNN CLASSIFICATION\n');
fprintf('============================================\n');

fprintf('Predicted Class : %s\n', string(resultLabel));

for k = 1:numel(classNames)

    fprintf('%-15s : %.4f %%\n', ...
        string(classNames(k)), scores(1,k) * 100);

end

fprintf('\n============================================\n');
fprintf('          PIPELINE COMPLETE\n');
fprintf('============================================\n');
