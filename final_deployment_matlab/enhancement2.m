function enhanced_img = enhancement2(bmode)

% Processing chain:
% B-mode -> Median Filter or Weiner Filter -> Bilateral Filter
%skip adaptive filter it gives a inverse effect.

%% Normalize input
img = mat2gray(bmode);
img = medfilt2(img, [3 3]);
img = imbilatfilt(img);

%% Convert to dB
enhanced_img = 20 * log10(img + eps);

end