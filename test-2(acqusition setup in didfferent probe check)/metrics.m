

clc;
clear;
close all;

%% ============================================================
% PICMUS ULTRASOUND PIPELINE
% EXECUTION TIME + THROUGHPUT + IMAGE QUALITY BENCHMARK
%% ============================================================

total_timer = tic;

%% 1. LOAD ULTRASOUND DATA

t = tic;

data = load("/MATLAB Drive/mathwork_competition/test-2(acqusition setup in didfferent probe check)/PICMUS_RF_1.mat");

sig = squeeze(data.rf);

time_data_loading = toc(t);

%% 2. LOAD HEADER

t = tic;

header = data;

time_header = toc(t);

%% 3. RECONSTRUCTION PARAMETERS

t = tic;

param.x = linspace(-0.0125, 0.0125, 128);
param.y = linspace(0, 0.0725, 500);

param.fs = 20832000;
param.c = 1540;
param.p = 0;

param.TXangle = deg2rad(header.angles);
param.pitch = 3e-4;

param.passband = [0.3, 0.7];
param.ncol = 128;
param.to = 0;

time_parameters = toc(t);

%% 4. PREPROCESSING

t = tic;

useZeroDegreeOnly = false;

if useZeroDegreeOnly
    [sig, param] = preprocessing(sig, header, param, 1);
else
    [sig, param] = preprocessing(sig, header, param);
end

time_preprocessing = toc(t);

%% 5. FK MIGRATION

t = tic;

[migSIG, param] = fkmig(sig, param);

time_fkmig = toc(t);

%% 6. ENVELOPE DETECTION

t = tic;

env = abs(hilbert(migSIG));

time_envelope = toc(t);

%% 7. NORMALIZATION

t = tic;

env = env ./ (max(env(:)) + eps);

time_normalization = toc(t);

%% 8. GAIN ENHANCEMENT

t = tic;

gain_dB = 10;

gain = 10^(gain_dB / 20);

env_enhanced = env * gain;

time_gain = toc(t);

%% 9. LOG COMPRESSION

t = tic;

env_dB = 20 * log10(env_enhanced + eps);

time_log = toc(t);

%% 10. FOUR-STAGE IMAGE ENHANCEMENT

t = tic;

enhanced_img = enhancement2(env_enhanced);

enhanced_img = squeeze(enhanced_img);

enhanced_img = single(enhanced_img);

time_enhancement = toc(t);

%% ============================================================
% 11. IMAGE QUALITY EVALUATION
%% ============================================================

t = tic;

% ROI format:
% [x_start x_end y_start y_end]
%
% These are example coordinates.
% Select actual lesion and homogeneous background ROIs
% appropriate for the PICMUS image.

signal_roi = [45 75 180 230];

background_roi = [5 30 180 230];

% Ensure both images have identical dimensions

if ~isequal(size(env), size(enhanced_img))
    error("FK and enhanced images have different dimensions.");
end

[image_height, image_width] = size(env);

% Validate ROI coordinates

if signal_roi(1) < 1 || signal_roi(2) > image_width || ...
   signal_roi(3) < 1 || signal_roi(4) > image_height

    error("Signal ROI is outside image dimensions.");

end

if background_roi(1) < 1 || background_roi(2) > image_width || ...
   background_roi(3) < 1 || background_roi(4) > image_height

    error("Background ROI is outside image dimensions.");

end

% FK image in linear amplitude

fk_image = env;

% enhancement2 performs mat2gray, filtering and dB conversion.
% Convert enhanced output back to linear amplitude for metrics.

enh_image = 10.^(double(enhanced_img) / 20);

% Extract FK ROIs

signal_fk = fk_image( ...
    signal_roi(3):signal_roi(4), ...
    signal_roi(1):signal_roi(2));

background_fk = fk_image( ...
    background_roi(3):background_roi(4), ...
    background_roi(1):background_roi(2));

% Extract enhanced ROIs

signal_enh = enh_image( ...
    signal_roi(3):signal_roi(4), ...
    signal_roi(1):signal_roi(2));

background_enh = enh_image( ...
    background_roi(3):background_roi(4), ...
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

%% ============================================================
% 12. ACQUISITION LATENCY AND THROUGHPUT
%% ============================================================

% Processing latency excludes file loading and setup.
% Includes preprocessing through enhancement.

acquisition_latency = ...
    time_preprocessing + ...
    time_fkmig + ...
    time_envelope + ...
    time_normalization + ...
    time_gain + ...
    time_log + ...
    time_enhancement;

% Single acquisition: all statistics are identical.

latency_samples = acquisition_latency;

mean_latency = mean(latency_samples);

min_latency = min(latency_samples);

max_latency = max(latency_samples);

% Estimated throughput

throughput = 1 / mean_latency;

%% 13. TOTAL EXECUTION TIME

total_execution_time = toc(total_timer);

%% ============================================================
% 14. PRINT EXECUTION TIMES
%% ============================================================

fprintf('\n============================================\n');
fprintf('       PICMUS MATLAB EXECUTION TIMES\n');
fprintf('============================================\n');

fprintf('Data Loading               : %.6f sec\n', time_data_loading);
fprintf('Header Loading             : %.6f sec\n', time_header);
fprintf('Parameter Setup            : %.6f sec\n', time_parameters);
fprintf('Preprocessing              : %.6f sec\n', time_preprocessing);
fprintf('FK Migration               : %.6f sec\n', time_fkmig);
fprintf('Envelope Detection         : %.6f sec\n', time_envelope);
fprintf('Normalization              : %.6f sec\n', time_normalization);
fprintf('Gain Enhancement           : %.6f sec\n', time_gain);
fprintf('Log Compression            : %.6f sec\n', time_log);
fprintf('Four-Stage Enhancement     : %.6f sec\n', time_enhancement);
fprintf('Image Quality Evaluation   : %.6f sec\n', time_quality);

fprintf('--------------------------------------------\n');
fprintf('TOTAL EXECUTION TIME       : %.6f sec\n', total_execution_time);

%% ============================================================
% 15. PERFORMANCE BENCHMARK
%% ============================================================

fprintf('\n============================================\n');
fprintf('          PERFORMANCE BENCHMARK\n');
fprintf('============================================\n');

fprintf('Mean Latency       : %.6f sec\n', mean_latency);

fprintf('Minimum Latency    : %.6f sec\n', min_latency);

fprintf('Maximum Latency    : %.6f sec\n', max_latency);

fprintf('Throughput         : %.4f acquisitions/sec\n', throughput);

fprintf('Total Execution    : %.6f sec\n', total_execution_time);

fprintf('============================================\n');

%% ============================================================
% 16. IMAGE QUALITY METRICS
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
% 17. DISPLAY FK MIGRATED IMAGE
%% ============================================================

% Use param.z when its length matches the image height.
% Otherwise use param.y if it matches.

if isfield(param, 'z') && numel(param.z) == image_height

    depth_axis = param.z;

elseif isfield(param, 'y') && numel(param.y) == image_height

    depth_axis = param.y;

else

    error("No matching depth coordinate vector found.");

end

figure;

imagesc(param.x * 1000, depth_axis * 1000, env_dB);

axis image;

colormap gray;

caxis([-60 0]);

ylim([0 35]);

colorbar;

xlabel('Lateral position (mm)');

ylabel('Depth (mm)');

title('PICMUS F-k Migrated +10 dB Gain');

%% ============================================================
% 18. DISPLAY ENHANCED IMAGE
%% ============================================================

figure;

imagesc(param.x * 1000, depth_axis * 1000, enhanced_img);

axis image;

colormap gray;

caxis([-60 0]);

ylim([0 35]);

colorbar;

xlabel('Lateral position (mm)');

ylabel('Depth (mm)');

title('PICMUS Enhanced B-mode');

%% ============================================================
% 19. PIPELINE COMPLETE
%% ============================================================

fprintf('\n============================================\n');
fprintf('          PICMUS PIPELINE COMPLETE\n');
fprintf('============================================\n');
