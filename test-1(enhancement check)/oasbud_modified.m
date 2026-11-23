

%% Load OASBUD dataset
load('/MATLAB Drive/OASBUD (1).mat');

%% Parameters
c = 1540;       % Speed of sound (m/s)
width = 38;     % Aperture width (mm)
fs = 40e6;      % Sampling frequency (Hz)

%% Load RF data
rf = data(1).rf1;

%% Axes
z_axis = 1000 * linspace(0, size(rf, 1)*0.5*c/fs, size(rf, 1));
y_axis = linspace(0, width, size(rf, 2));

%% Envelope detection
% For this version, use RF data directly as the input signal.
migSIG = rf;

env = abs(hilbert(migSIG));

%% Normalize
env = env ./ (max(env(:)) + eps);

%% Uniform +10 dB Gain
gain_dB = 10;
gain = 10^(gain_dB / 20);

env_enhanced = env * gain;

%% Log compression
env_dB = 20 * log10(env_enhanced + eps);

%% ROI masking
envelope_image = env_dB;

roi = logical(data(1).roi1);

% Set background outside ROI to minimum display intensity
envelope_image(~roi) = -60;

%% Visualization
figure;

imagesc(y_axis, z_axis, envelope_image);

colormap(gray);
xlabel('mm');
ylabel('mm');

set(gca, 'CLim', [-60 10]);
set(gca, 'FontSize', 14);

axis image;