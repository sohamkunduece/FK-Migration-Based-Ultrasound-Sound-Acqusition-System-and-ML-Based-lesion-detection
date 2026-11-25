
function [sig, param] = preprocessing(sig, USHEADER, param, n_ang_ss)
% PREPROCESSING
% Preprocesses raw ultrasound acquisition data before FK migration.
%
% Inputs:
%   sig       : Ultrasound data [time x channels x angles]
%   USHEADER  : Ultrasound acquisition header structure
%   param     : FK migration parameter structure
%   n_ang_ss  : Number of transmission angles to retain
%               [] or omitted = all angles
%               1 = angle closest to zero degrees
%               >1 = evenly subsampled angles
%
% Outputs:
%   sig       : Corrected ultrasound acquisition data
%   param     : Updated FK migration parameters
%
% Corrections:
%   1. Lens delay correction
%   2. Angle-dependent delay correction
%   3. Channel 126 amplitude correction
%   4. Optional transmission angle subsampling

    %% Default settings

    if nargin < 4 || isempty(n_ang_ss)
        n_ang_ss = 0;  % Retain all angles
    end

    lens_delay = 96;

    %% Ensure data has three dimensions

    sig = squeeze(sig);

    if ndims(sig) == 2
        sig = reshape(sig, size(sig,1), size(sig,2), 1);
    end

    sig = double(sig);

    %% Load acquisition parameters
    %%changed here due yo oading type. see readme.
    tx_angles = double(USHEADER.angles(:));

    n_ele = size(sig, 2);
    n_angles = size(sig, 3);

    fs = double(param.fs);
    c = double(param.c);
    pitch = double(param.pitch);

    % Check dimensions

    if length(tx_angles) ~= n_angles
        error(['Number of transmission angles does not match ', ...
               'the third dimension of ultrasound data.']);
    end

    %% Angle subsampling

    if n_ang_ss == 1

        % Select angle closest to zero degrees

        [~, ang_ind] = min(abs(tx_angles));

    elseif n_ang_ss > 1 && n_ang_ss < n_angles

        % Select evenly distributed transmission angles

        ang_ss = linspace(tx_angles(1), tx_angles(end), n_ang_ss);

        ang_ind = zeros(1, n_ang_ss);

        for k = 1:n_ang_ss
            [~, ang_ind(k)] = min(abs(tx_angles - ang_ss(k)));
        end

    elseif n_ang_ss == 0 || n_ang_ss == n_angles

        % Retain all transmission angles

        ang_ind = 1:n_angles;

    else

        error('Invalid number of subsampled angles.');

    end

    % Retain selected angles

    sig = sig(:, :, ang_ind);
    tx_angles = tx_angles(ang_ind);

    %% 1. Lens delay correction

    if size(sig, 1) <= lens_delay
        error('Insufficient signal samples for lens delay correction.');
    end

    sig = sig(lens_delay+1:end, :, :);

    %% 2. Angle-dependent delay correction

    n_t = size(sig, 1);

    dt = 1 / fs;

    delay = abs( ...
        ((n_ele - 1) / 2) .* pitch .* ...
        sin(deg2rad(tx_angles)) ./ c ...
    );

    delay_ind = floor(delay ./ dt);

    for k = 1:length(tx_angles)

        d = delay_ind(k);

        if d == 0
            continue;
        end

        if d >= n_t
            sig(:, :, k) = 0;
            continue;
        end

        % Shift signals to correct transmission delay

        sig(1:n_t-d, :, k) = sig(d+1:n_t, :, k);

        % Zero-pad the end

        sig(n_t-d+1:n_t, :, k) = 0;

    end

    %% 3. Channel 126 correction
    % Python channel index 125 corresponds to MATLAB index 126.

    if size(sig, 2) >= 126
        sig(:, 126, :) = 2 .* sig(:, 126, :);
    else
        warning('Channel 126 does not exist. Skipping channel correction.');
    end

    %% 4. Update FK migration parameters

    param.TXangle = deg2rad(tx_angles);

    fprintf('\nUltrasound preprocessing completed.\n');
    fprintf('Remaining samples: %d\n', size(sig, 1));
    fprintf('Number of channels: %d\n', size(sig, 2));
    fprintf('Number of transmission angles: %d\n', size(sig, 3));

end
