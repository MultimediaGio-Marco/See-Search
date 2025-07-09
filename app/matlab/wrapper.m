function wrapper()
  I = imread('input.jpg');                 % carica immagine
  [BW, mag, den] = edge_pipeline(I);       % chiama la tua funzione
  imwrite(BW, 'output.png');               % salva il risultato
end
