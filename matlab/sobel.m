function [Imagedx,Imagedy] = sobel(Image)
    keneldx=[-1,0,1
             -2,0,2
             -1,0,1];

    keneldy=[-1,-2,-1
              0,0,0
              1,2,1];

    Imagedx = imfilter(Image, keneldx, 'symmetric', 'same');
    Imagedy = imfilter(Image, keneldy, 'symmetric', 'same');

end