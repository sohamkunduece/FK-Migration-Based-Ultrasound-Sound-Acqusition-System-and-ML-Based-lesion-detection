
clc;
clear;
close all;

%% TOTAL EXECUTION TIMER
total_timer = tic;

%% 1. Load Ultrasound Data
t = tic;

data = load("/MATLAB Drive/dataset/Hypo_lesion/USDATA_20210624112717871.mat");
sig = squeeze(data.USDATA);

time_data_load = toc(t);

%% 2. Load Header
t = tic;

header = load("/MATLAB Drive/mathwork_competition/Acquisition_setup/USHEADER_20210624112712.mat");

time_header_load = toc(t);

%% 3. Reconstruction Parameters
t = tic;

param.x = linspace(-0.0125, 0.0125, 128);
param.y = linspace(0, 0.0725, 500);

param.fs = 31250000;
param.c = 1540;
param.p = 0;

param.TXangle = deg2rad(header.USHEADER.xmitAngles);
param.pitch = 1.953000000000000e-04;

param.passband = [0.3, 0.7];
param.ncol = 128;
param.to = 0;

time_parameters = toc(t);

%% 4. Preprocessing
t = tic;

[sig, param] = preprocessing(sig, header.USHEADER, param);

time_preprocessing = toc(t);

%% 5. FK Migration
t = tic;

[migSIG, param] = fkmig(sig, param);

time_fkmig = toc(t);

%% 6. Envelope Detection
t = tic;

env = abs(hilbert(migSIG));

time_envelope = toc(t);

%% 7. Normalization
t = tic;

env = env ./ (max(env(:)) + eps);

time_normalization = toc(t);

%% 8. Gain Enhancement
t = tic;

gain_dB = 10;
gain = 10^(gain_dB / 20);

env_enhanced = env * gain;

time_gain = toc(t);

%% 9. Log Compression
t = tic;

env_dB = 20 * log10(env_enhanced + eps);

time_log_compression = toc(t);

%% 10. Four-Stage Enhancement
t = tic;

enhanced_img = enhancement2(env_enhanced);

time_enhancement = toc(t);

%% 11. IMAGE QUALITY EVALUATION

t = tic;

% Use linear envelope images for quality calculations
% Do not calculate SNR directly from log-compressed data.

fk_image = env;
enhanced_image = enhanced_img;

% IMPORTANT:
% Define signal and background regions using pixel coordinates.
% Format: [x_start x_end y_start y_end]
%
% Replace these example coordinates after inspecting your image.
% Choose a lesion ROI for signal and a homogeneous background ROI
% for noise, avoiding image boundaries.

signal_roi = [45 75 180 230];
background_roi = [5 30 180 230];

% Extract ROIs
signal_fk = fk_image(signal_roi(3):signal_roi(4), ...
                     signal_roi(1):signal_roi(2));

noise_fk = fk_image(background_roi(3):background_roi(4), ...
                     background_roi(1):background_roi(2));

signal_enh = enhanced_image(signal_roi(3):signal_roi(4), ...
                            signal_roi(1):signal_roi(2));

noise_enh = enhanced_image(background_roi(3):background_roi(4), ...
                           background_roi(1):background_roi(2));

% SNR calculation
% Mean signal power / background noise power
snr_fk = 10 * log10(mean(signal_fk(:).^2) / ...
                    (var(noise_fk(:)) + eps));

snr_enh = 10 * log10(mean(signal_enh(:).^2) / ...
                     (var(noise_enh(:)) + eps));

% Contrast-to-Noise Ratio (CNR)
cnr_fk = abs(mean(signal_fk(:)) - mean(noise_fk(:))) / ...
         (sqrt(var(signal_fk(:)) + var(noise_fk(:)))/2 + eps);

cnr_enh = abs(mean(signal_enh(:)) - mean(noise_enh(:))) / ...
          (sqrt(var(signal_enh(:)) + var(noise_enh(:)))/2 + eps);

% Equivalent Number of Looks (ENL)
% Calculate in the homogeneous background ROI
enl_fk = mean(noise_fk(:))^2 / (var(noise_fk(:)) + eps);

enl_enh = mean(noise_enh(:))^2 / (var(noise_enh(:)) + eps);

% Mean intensity
mean_fk = mean(fk_image(:));
mean_enh = mean(enhanced_image(:));

% Standard deviation
std_fk = std(fk_image(:));
std_enh = std(enhanced_image(:));

% Dynamic range
dynamic_fk = max(fk_image(:)) - min(fk_image(:));
dynamic_enh = max(enhanced_image(:)) - min(enhanced_image(:));

time_quality = toc(t);

%% 12. TOTAL EXECUTION TIME
total_time = toc(total_timer);

%% 13. PRINT RESULTS

fprintf('\n============================================\n');
fprintf('       MATLAB ONLINE EXECUTION TIMES\n');
fprintf('============================================\n');

fprintf('Data Loading             : %.6f sec\n', time_data_load);
fprintf('Header Loading           : %.6f sec\n', time_header_load);
fprintf('Parameter Setup          : %.6f sec\n', time_parameters);
fprintf('Preprocessing            : %.6f sec\n', time_preprocessing);
fprintf('FK Migration              : %.6f sec\n', time_fkmig);
fprintf('Envelope Detection       : %.6f sec\n', time_envelope);
fprintf('Normalization            : %.6f sec\n', time_normalization);
fprintf('Gain Enhancement         : %.6f sec\n', time_gain);
fprintf('Log Compression          : %.6f sec\n', time_log_compression);
fprintf('Four-Stage Enhancement   : %.6f sec\n', time_enhancement);
fprintf('Image Quality Evaluation : %.6f sec\n', time_quality);

fprintf('--------------------------------------------\n');
fprintf('TOTAL TIME               : %.6f sec\n', total_time);

fprintf('\n============================================\n');
fprintf('          IMAGE QUALITY METRICS\n');
fprintf('============================================\n');

fprintf('\n--- FK MIGRATED IMAGE ---\n');
fprintf('SNR       : %.4f dB\n', snr_fk);
fprintf('CNR       : %.6f\n', cnr_fk);
fprintf('ENL       : %.6f\n', enl_fk);
fprintf('Mean      : %.6f\n', mean_fk);
fprintf('Std Dev   : %.6f\n', std_fk);
fprintf('Dynamic Range : %.6f\n', dynamic_fk);

fprintf('\n--- ENHANCED IMAGE ---\n');
fprintf('SNR       : %.4f dB\n', snr_enh);
fprintf('CNR       : %.6f\n', cnr_enh);
fprintf('ENL       : %.6f\n', enl_enh);
fprintf('Mean      : %.6f\n', mean_enh);
fprintf('Std Dev   : %.6f\n', std_enh);
fprintf('Dynamic Range : %.6f\n', dynamic_enh);

fprintf('\n============================================\n');
fprintf('          PERFORMANCE COMPARISON\n');
fprintf('============================================\n');

fprintf('SNR Difference : %.4f dB\n', snr_enh - snr_fk);
fprintf('CNR Difference : %.6f\n', cnr_enh - cnr_fk);
fprintf('ENL Difference : %.6f\n', enl_enh - enl_fk);

fprintf('============================================\n');
