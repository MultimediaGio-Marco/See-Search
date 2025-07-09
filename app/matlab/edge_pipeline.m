% pipeLine
% RGB → YCbCr → estrai Y → GaussianBlur → Sobel(dx/dy) → magnitude → (Wavelet denoise) → soglia
function [BW_clean, magnitude, denoised] = edge_pipeline(imageRGB)

    % [1] RGB → YCbCr
    imageYCbCr = rgb2ycbcr(imageRGB);

    % [2] Estrai Y
    Y = double(imageYCbCr(:, :, 1));
    
    % [3] Gaussian Blur
    Y_blur = imgaussfilt(Y, 1); % sigma=1, puoi regolare

    [dx,dy]=sobel(Y_blur);
    lap = imfilter(Y_blur, fspecial('laplacian'));



    % [5] Magnitudo
    magnitude = sqrt(dx.^2 + dy.^2);
    magnitude = magnitude + abs(lap);
    magnitude = mat2gray(magnitude) * 255;
    

    % [6] Wavelet denoise
    % Richiede Wavelet Toolbox
    denoised = wdenoise2(magnitude, 2); % livello 2, regolabile

    % [7] Threshold (soglia fissa, regolabile)
    BW = denoised > 40;  % oppure: imbinarize(denoised)

    % Converte in immagine logica
    BW = logical(BW);

    se = strel('disk', 1);  % elemento strutturante
    BW_clean = imclose(BW, se);
    BW_clean = bwareaopen(BW_clean, 50);  % elimina aree < 50 pixel

end