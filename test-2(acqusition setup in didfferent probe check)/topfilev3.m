
clc;
clear;
close all;

%% Load ultrasound data

data = load("/MATLAB Drive/mathwork_competition/test-2(acqusition setup in didfferent probe check)/PICMUS_RF_2.mat");

sig = squeeze(data.rf);

%% Load header

header = data;

%% Reconstruction parameters

param.x = linspace(-0.0125, 0.0125, 128);
param.y = linspace(0, 0.0725, 500);

param.fs = 20832000;%%changed due to probe data
param.c = 1540;
param.p = 0;

param.TXangle = deg2rad(header.angles);
param.pitch = 3e-4;

param.passband = [0.3, 0.7];
param.ncol = 128;
param.to = 0;

%% PREPROCESSING

% Option 1: Retain all transmission angles
[sig, param] = preprocessing(sig, header, param);

% Option 2: Use only zero-degree transmission angle
% [sig, param] = preprocessing(sig, header.USHEADER, param, 1);

%% FK Migration

[migSIG, param] = fkmig(sig, param);

%% Envelope detection
env = abs(hilbert(migSIG));

%% Normalize
env = env ./ (max(env(:)) + eps);

%% Uniform +6 dB Gain
gain_dB = 10;
gain = 10^(gain_dB / 20);

env_enhanced = env * gain;

%% Log compression
env_dB = 20 * log10(env_enhanced + eps);

%% Display
figure;
imagesc(param.x * 1000, param.z * 1000, env_dB);

axis image;
colormap gray;
caxis([-60 0]);
ylim([0 35]);
colorbar;

xlabel('Lateral position (mm)');
ylabel('Depth (mm)');
title('f-k Migrated + 6 dB Gain');

%% Four-stage Enhancement

enhanced_img = enhancement2(env_enhanced);


%% Display

figure;
imagesc(param.x * 1000, param.y * 1000, enhanced_img);

axis image;
colormap gray;
caxis([-60 0]);
ylim([0 35]);
colorbar;

xlabel('Lateral position (mm)');
ylabel('Depth (mm)');
title('Three-Stage Enhanced B-mode');

