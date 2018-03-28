% processPictures
%       Function that takes images from a given directory one by one and
%       saves them with a superpixel mask

% Parameters
%   originDir - Directory where the raw images are located
%   directory - Directory where the marked images are intended to be saved
%   iNumPixels - The number of superpixelsto be used in each image's mask

% Notes:
%   This function depends on the custom function called
%   "drawsuperpixelsonimage"



function processPictures(originDir, directory, iNumPixels)
    %Identify the images to process
    %files = dir('../Dataset/Rainy/*.jpg');
    %files = dir(sprintf('../Dataset/%s/*.jpg', category));
   
    access_files = strcat(originDir,'/*.jpg');
    files = dir(access_files);

    %Variable with the directory where images came from
    %originDir = input('What is the directory of the original images? ');
    %originDir = string('../Dataset/Rainy/');

    %Directory where the processed images will be saved
    %directory = input('Where should the processed images be saved (directory)?' );
    %directory = string('../Dataset/Rainy_SP/');

    %Number of desired pixels to add to the images
    %iNumPixels = input('How many superpixels should be added to each image? ');
    if ~isnumeric(iNumPixels)
      error('Error in the number of desired superpixels');
    end

    %counter to indicate the names of the images
    count = 1;
    for file = files'
        if mod(count,10) == 0
                disp(sprintf('[LOG] Processed image %s', file.name))
        end
        %Identifying the name of the current image
        aux = file.name;
        aux = strcat('/',aux);

        %Reading the file with the current image
        A = imread(strcat(originDir,aux));

        %Using the custom image that adds the pixels and saves the image
        drawsuperpixelsonimage(A,iNumPixels,count,directory);

        aux = '';
        count = count + 1;
    end
end