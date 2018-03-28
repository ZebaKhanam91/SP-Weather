%classifywather The purpose of this function is to classify the weather
%depicted in a given image

% Parameters
%   imagepath - is a sequence of chaacters containing the directory of the
%   image to be classified

% Output
%   result - a 2 x n matrix containing the pairs of category - score, where
%   n is the number of possible categories

function [result] = classifyweather(imagepath)

%   imagepath should contain the directory to a valid image

%   Apply Superpixel Mask - 50 SP

    % TO DO: set path to Matlab's caffe installation
    addpath('/home/zeba/caffe/matlab/')
    
    % Section taken from Matworks documentation on using superpixels
    % (From next line of code until comment containing *******)
    %   Mathworks, Inc. "2-D superpixel oversegmentation of images", Mathworks, 
    %   [Online]. Available: https://uk.mathworks.com/help/images/ref/superpixels.html. 
    %   [Accessed June 2017]
    
    % Load Image
    img = imread(imagepath);
    
    % Create Superpixel Mask
    [L,N] = superpixels(img, 50);
    BW = boundarymask(L);
    
    % Save Superpixel enhanced Image
    f = figure('visible', 'off');
    iptsetpref('ImshowBorder','tight');
    imshow(imoverlay(img,BW,'cyan'),'InitialMagnification','fit');
    saveas(f,'tempimage','jpg');
    % *******
    
    % Setting the model to use
    % TO DO: Set model
    model = 'ResNet50';
    
%   Extract features using the CNN Architecture

%   The following lines are from the matdeeprep function
%   (From next line of code until comment containing *******)
%       Matdeeprep: 
%       [1] G. Kalliatakis, S. Ehsan, M. Fasli, A. Leonardis, J. Gall and K. McDonald-Maier
%           Detection of Human Rights Violations in Images: Can Convolutional Neural Networks help?
%           Computer Vision, Imaging and Computer Graphics Theory and Applications, (VISAPP) Conference, 2017
    caffe.set_mode_cpu();
    
    model_dir = sprintf('/home/zeba/caffe/models/%s/', model); % Modified from Matdeeprep
    prototxt_file = dir( fullfile(model_dir,'*deploy.prototxt') );   %# list all *.prototxt files
    caffemodel_file = dir( fullfile(model_dir,'*.caffemodel') );   %# list all *.caffemodel files

    net_model = [model_dir prototxt_file.name]; %Complete path to the protoxt file found in the directory
    net_weights = [model_dir caffemodel_file.name]; %Complete path to the protoxt file found in the directory

    phase = 'test'; % run with phase test (so that dropout isn't applied)

    
    net = caffe.Net(net_model, net_weights, phase);
    
    % Some models require the dimensions of the images to be 224x224 while others require 227x227 
    if isequal(model,'bvlc_alexnet')==1 || isequal(model,'bvlc_reference_caffenet')==1 ||isequal(model,'bvlc_reference_rcnn_ilsvrc13')==1 ||isequal(model,'placesCNN')==1
        net.blobs('data').reshape([227 227 3 1]);
    else
        net.blobs('data').reshape([224 224 3 1]);
    end
    
    im = imread('tempimage.jpg');
    im = standardizeImage(im); % ensure of type single, w. three channels
    
    if isequal(model,'placesCNN')==1 
        mean_flag=1;
    else
        mean_flag=0;
    end
    if isequal(model,'bvlc_googlenet')==1 || isequal(model,'ResNet50')==1 ||isequal(model,'ResNet101')==1 ||isequal(model,'ResNet152')==1 ||isequal(model,'VGG_CNN_M')==1 ||isequal(model,'VGG_CNN_S')==1||isequal(model,'VGG_CNN_F')==1||isequal(model,'VGGNet16')==1 ||isequal(model,'VGGNet19')==1
    	input_data = {prepare_image_224_224(im,mean_flag)};
        temp=net.forward(input_data);
        
        % Check if ResNet is selected so that pool5 layer is called instead of the default fc7 for the other models or inception_5b/output for bvlc_googlenet
        if isequal(model,'ResNet50')==1 || isequal(model,'ResNet101') || isequal(model,'ResNet152')
            data = net.blobs('res5c_branch2c').get_data;
            data = data(2:2:6, 2:2:6, : );
            
            code_v(:,1) = data( : );
            
        elseif isequal(model,'VGG_CNN_M')==1 || isequal(model,'VGGNet16')==1 || isequal(model,'VGGNet19')==1 || isequal(model,'VGG_CNN_S')==1 || isequal(model,'VGG_CNN_F')==1
            code_v(:,1) = net.blobs('fc7').get_data();
            
        elseif isequal(model,'bvlc_googlenet')==1 
            data = net.blobs('inception_5b/output').get_data;
            data = data(2:2:6, 2:2:6, : );
            code_v(:,1) = data( : );
            
        end
        
    else
    	input_data = {prepare_image(im,mean_flag)};
        temp=net.forward(input_data);
        code_v(:,1) = net.blobs('fc7').get_data();
        
    end
    % *******
    


%   Load classifiers weights

    % TO DO: Add path to file with the classifier to be used
    file_contents = load('../Classifiers/50SP/ResNet50.mat');
    categories = {};
    weights = {};
    bias = {};
    
    % Extracting all the categories and their corresponding weights/bias
    for k = 1:(numel(file_contents.classifiers)/3)
       categories = [categories file_contents.classifiers{3 * k - 2}];
       weights = [weights file_contents.classifiers{3 * k - 1}];
       bias = [bias file_contents.classifiers{3 * k}];
    end

    %   Get Classifiers Scores
    scores = []; % Variable to store the calculated scores
    cats = []; % Variable to store the categories considered
    
        % Auxiliary variables
    first_score = -100;
    first_cat = '';
    second_score = -50;
    second_cat = '';
    
    result = []; % Output variable containing the classification results
    
    %   Calculate a score for each of the categories and keep track of the
    %   best and second best scores and they respective categories
    for k = 1:(numel(file_contents.classifiers)/3)
        
       % Calculating the score with the weights, extracted features and
       % bias
       aux = weights{k}' * code_v + bias{k};
       
       % Keep track of the results
       cats = [cats string(categories{k})];
       scores = [scores aux];
       
       % Update highest scores
       if aux > first_score
          second_cat = first_cat;
          second_score = first_score;
          first_cat = categories{k};
          first_score = aux;
       elseif aux > second_score
          second_cat = categories{k};
          second_score = aux;
       end
    end
    
    % Set value for the output variable
    result = vertcat(cats, scores);

    %   Interpret Classifiers Scores and Display Results
    disp(sprintf('The most likely weather in image %s is %s, but it also appears to have features of %s.', imagepath,first_cat, second_cat));

end


%   The following lines of code are from the matdeeprep source code and 
%   are useful for its use in this file
% (From next line of code until comment containing *******)
%       Matdeeprep: 
%       [1] G. Kalliatakis, S. Ehsan, M. Fasli, A. Leonardis, J. Gall and K. McDonald-Maier
%           Detection of Human Rights Violations in Images: Can Convolutional Neural Networks help?
%           Computer Vision, Imaging and Computer Graphics Theory and Applications, (VISAPP) Conference, 2017
function cropped_data = prepare_image_224_224(im,mean_flag)
    if mean_flag==1
        d = load('/home/zeba/caffe/models/placesCNN/places_mean.mat');
        mean_data = d.image_mean;
    else
        d = load('/home/zeba/caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
        mean_data = d.mean_data;
    end
    IMAGE_DIM = 256;
    % Convert an image returned by Matlab's imread to im_data in caffe's data format: W x H x C with BGR channels
    im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
    im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)
    cropped_data = im_data(15:238, 15:238, :); % take 224 x 224 center crop
end

function cropped_data = prepare_image(im,mean_flag)
    if mean_flag==1
        d = load('/home/zeba/caffe/models/placesCNN/places_mean.mat');
        mean_data = d.image_mean;
    else
        d = load('/home/zeba/caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
        mean_data = d.mean_data;
    end
    IMAGE_DIM = 256;
    % Convert an image returned by Matlab's imread to im_data in caffe's data format: W x H x C with BGR channels
    im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
    im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)
    cropped_data = im_data(15:241, 15:241, :); % take 227 x 227 center crop
end
% *******)
