% drawsuperpixelsonimage
%       Function that takes an image from a path, calculates superpixels,
%       creates superpixel mask and saves the image with the superpixel
%       mask

% Parameters
%   imageFile - Image file loaded as Matlab object
%   iNumPixels - The number of superpixelsto be used in each image's mask
%   count - Number of current image
%   directory - path to location where modified image is expected to be
%               saved

function drawsuperpixelsonimage( imageFile, iNumPixels, count, directory,col)

    %   The following lines of code are from MATLAB's documentation
    %   source code with slight modifications
    %   (From next line of code until comment containing *******)
    %   Source:
    %       Mathworks Inc. "2-D superpixel oversegmentation of images".
    %       Mathworks, [Online]. 
    %       Available: https://uk.mathworks.com/help/images/ref/superpixels.html. 
    %       [Accessed June 2017].
    
    A =imageFile;
    [L,N] = superpixels(A,iNumPixels);
    BW = boundarymask(L);
    % *******
    directory = strcat(directory,'\');
%      final = char(strcat(directory,pad(string(count),4,'left','0')))
    
    %figure
    %iptsetpref('ImshowBorder','tight');
    I = imoverlay(A,BW,col);
   %imshow(imoverlay(A,BW,'cyan'),'InitialMagnification','fit')
   imwrite(I,char(strcat(directory,pad(string(count),4,'left','0'),'.jpg')));
  % saveas(f,char(strcat(directory,pad(string(count),4,'left','0'))),'jpg')
end

