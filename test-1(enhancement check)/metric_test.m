
clc;
close all;
clear;

%% ============================================================
% LOAD OASBUD DATASET
% =============================================================

load('/MATLAB Drive/OASBUD (1).mat');

%% Parameters
c = 1540;
width = 38;
fs = 40e6;

%% RF Data
rf = data(1).rf1;
roi = logical(data(1).roi1);

%% Axes
z_axis = 1000 * linspace(0, size(rf,1)*0.5*c/fs, size(rf,1));
y_axis = linspace(0, width, size(rf,2));

%% ============================================================
% METHOD 1: CREATOR'S ENVELOPE DETECTION
% =============================================================

creator_env = abs(hilbert(rf));

% Original log compression
creator_dB = 20 * log10(creator_env + eps);

% Original creator display image
creator_original = creator_dB;
creator_original(~roi) = NaN;

%% ============================================================
% METHOD 2: YOUR ENVELOPE DETECTION
% =============================================================

% Envelope detection
env = abs(hilbert(rf));

% Normalize
env = env ./ (max(env(:)) + eps);

% Uniform +10 dB gain
gain_dB = 10;
gain = 10^(gain_dB/20);

env_enhanced = env * gain;

% Log compression
your_dB = 20 * log10(env_enhanced + eps);

% ROI masking
your_original = your_dB;
your_original(~roi) = NaN;

%% ============================================================
% ALIGN INTENSITY REFERENCES FOR FAIR COMPARISON
% =============================================================

% Use the same global maximum reference
creator_aligned = creator_dB - max(creator_dB(:)) + 10;

your_aligned = your_dB;

% Apply identical ROI
creator_aligned(~roi) = NaN;
your_aligned(~roi) = NaN;

% Extract ROI pixels
A = creator_aligned(roi);
B = your_aligned(roi);

%% ============================================================
% METRICS
% =============================================================

% Basic image statistics
creator_mean = mean(A);
your_mean = mean(B);

creator_std = std(A);
your_std = std(B);

creator_min = min(A);
your_min = min(B);

creator_max = max(A);
your_max = max(B);

creator_range = creator_max - creator_min;
your_range = your_max - your_min;

% Contrast (standard deviation / mean linear envelope)
creator_contrast = std(creator_env(roi)) / ...
                   (mean(creator_env(roi)) + eps);

your_contrast = std(env_enhanced(roi)) / ...
                (mean(env_enhanced(roi)) + eps);

% Image difference metrics
difference = A - B;

RMSE = sqrt(mean(difference.^2));
MAE = mean(abs(difference));

% Correlation
C = corrcoef(A, B);
correlation = C(1,2);

% PSNR (70 dB comparison range)
peak_value = 70;
MSE = mean(difference.^2);

PSNR = 10 * log10(peak_value^2 / (MSE + eps));

%% ============================================================
% DISPLAY METRICS
% =============================================================

fprintf('\n============================================\n');
fprintf('       OASBUD ENVELOPE COMPARISON\n');
fprintf('============================================\n');

fprintf('\n--- IMAGE STATISTICS (ALIGNED dB) ---\n');

fprintf('\nCreator:\n');
fprintf('Mean intensity       : %.4f dB\n', creator_mean);
fprintf('Standard deviation   : %.4f dB\n', creator_std);
fprintf('Minimum intensity    : %.4f dB\n', creator_min);
fprintf('Maximum intensity    : %.4f dB\n', creator_max);
fprintf('Dynamic range        : %.4f dB\n', creator_range);
fprintf('Linear envelope contrast: %.4f\n', creator_contrast);

fprintf('\nYour Method:\n');
fprintf('Mean intensity       : %.4f dB\n', your_mean);
fprintf('Standard deviation   : %.4f dB\n', your_std);
fprintf('Minimum intensity    : %.4f dB\n', your_min);
fprintf('Maximum intensity    : %.4f dB\n', your_max);
fprintf('Dynamic range        : %.4f dB\n', your_range);
fprintf('Linear envelope contrast: %.4f\n', your_contrast);

fprintf('\n--- IMAGE SIMILARITY ---\n');

fprintf('RMSE                 : %.8f dB\n', RMSE);
fprintf('MAE                  : %.8f dB\n', MAE);
fprintf('Correlation          : %.8f\n', correlation);
fprintf('PSNR                 : %.4f dB\n', PSNR);

fprintf('\n--- DATASET INFORMATION ---\n');

fprintf('RF dimensions        : %d x %d\n', size(rf,1), size(rf,2));
fprintf('ROI pixels           : %d\n', sum(roi(:)));
fprintf('ROI coverage         : %.2f %%\n', ...
    100*sum(roi(:))/numel(roi));

fprintf('\n============================================\n');

%% ============================================================
% VISUAL COMPARISON
% =============================================================

figure('Name','Envelope Comparison','Color','w');

subplot(1,2,1);

imagesc(y_axis,z_axis,creator_original);
colormap(gray);
clim([40 80]);
axis image;
set(gca,'FontSize',12);
title('Creator Original');
xlabel('mm');
ylabel('mm');

subplot(1,2,2);

imagesc(y_axis,z_axis,your_original);
colormap(gray);
clim([-60 10]);
axis image;
set(gca,'FontSize',12);
title('Your Original');
xlabel('mm');
ylabel('mm');

%% Aligned comparison
figure('Name','Aligned Envelope Comparison','Color','w');

subplot(1,2,1);

imagesc(y_axis,z_axis,creator_aligned);
colormap(gray);
clim([-60 10]);
axis image;
title('Creator (Aligned)');
xlabel('mm');
ylabel('mm');

subplot(1,2,2);

imagesc(y_axis,z_axis,your_aligned);
colormap(gray);
clim([-60 10]);
axis image;
title('Your Method (Aligned)');
xlabel('mm');
ylabel('mm');

%% Difference image
figure('Name','Difference Image','Color','w');

difference_image = creator_aligned - your_aligned;

imagesc(y_axis,z_axis,difference_image);
colormap(gray);
colorbar;
clim([-1 1]);
axis image;

title('Difference Image (dB)');
xlabel('mm');
ylabel('mm');
