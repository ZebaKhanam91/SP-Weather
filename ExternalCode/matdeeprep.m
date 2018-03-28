%% MatDeepRep ??? Deep representation learning tool for Image Classification using Trasnfer Learning, the BVLC Caffe Matlab interface (matcaffe) & various pretrained .caffemodel binaries
% 
%   ??? ??? ??? ??? ??? ??? ??? ??? ??? ???  ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ???
%   ???                                                                          ???
%   ???       MatDeepRep is a Matlab tool, built on top of Caffe framework,      ???
%   ???       capable of learning general deep feature representations           ???
%   ???       for image classification using pre-trained Deep ConvNet Models     ???
%   ???                                                                          ???
%   ???       Version 1.0 (8 September 2016 ) Initial release                    ???
%   ???                                                                          ???
%   ???       Author: ??Grigorios Kalliatakis (gkallia@essex.ac.uk)               ???
%   ???       Homepage: gkalliatakis.com                                         ???
%   ???       Embedded and Intelligent Systems Laboratory (EIS),                 ???
%   ???       University of Essex,UK                                             ???
%   ???                                                                          ??? 
%   ??? ??? ??? ??? ??? ??? ??? ??? ??? ???  ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ??? ???

function [code,code_v,code_neg,code_v_neg] = matdeeprep(model,dataset,category)
% ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
% Usage Example:  [code,code_v] = matdeeprep('ResNet50','FMD','fabric');
% ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

% Possible Settings for Inputs:

% [1] model =
%              'bvlc_alexnet' (AlexNet trained on ILSVRC 2012, almost exactly as described in "ImageNet classification with deep convolutional neural networks" dim: 227x227)

%              'bvlc_googlenet' (GoogLeNet trained on ILSVRC 2012, almost exactly as described in "Going Deeper with Convolutions"  dim: 224x224)

%              'bvlc_reference_caffenet' (AlexNet trained on ILSVRC 2012, with a minor variation from the version as described in ImageNet classification with deep convolutional neural networks dim: 227x227)

%              'bvlc_reference_rcnn_ilsvrc13' (pure Caffe implementation of Region-based Convolutional Neural Networks-R-CNN dim: 227x227)

%              'placesCNN' (AlexNet CNN trained on 205 scene categories of Places Database with  2.5 million images as described in "Learning Deep Features for Scene Recognition using Places Database" dim: 227x227)

%              'ResNet50'  (50-layer ResNet as described in "Deep Residual Learning for Image Recognition" dim: 224x224)

%              'ResNet101'  (101-layer ResNet as described in "Deep Residual Learning for Image Recognition" dim: 224x224)

%              'ResNet152'  (152-layer ResNet as described in "Deep Residual Learning for Image Recognition" dim: 224x224)

%              'VGG_CNN_F'  (CNN_F trained on the ILSVRC-2012, as described in "Return of the Devil in the Details: Delving Deep into Convolutional Nets"  dim: 224x224)

%              'VGG_CNN_M'  (CNN_M trained on the ILSVRC-2012, as described in "Return of the Devil in the Details: Delving Deep into Convolutional Nets"  dim: 224x224)

%              'VGG_CNN_S'  (CNN_S trained on the ILSVRC-2012, as described in "Return of the Devil in the Details: Delving Deep into Convolutional Nets"  dim: 224x224)

%              'VGGNet16'  (16-layer VGG-Net as described in "Very deep convolutional networks for large-scale image recognition" dim: 224x224)

%              'VGGNet19'  (19-layer VGG-Net as described in "Very deep convolutional networks for large-scale image recognition" dim: 224x224)

% Outputs:

%(1)     code   = the extracted features for the positive training images
%(2)     code_v = the extracted features for the positive test images
%(3)     code_neg   = the extracted features for the negative training images
%(4)     code_v_neg = the extracted features for the negative test images

%% (1) Network initialisation
%Adding location for MATLAB's caffe 
addpath('/Users/JoseCarlosVillarreal/Documents/caffe/matlab/')

initial_test_img_num = 770;

% Initialising code and code_v variables to ensure they are assigned and
% returned by the function
code = [];
code_neg = [];
code_v = [];
code_v_neg = [];

% Set caffe mode - At the moment only CPU is available
caffe.set_mode_cpu();

% Initialize the network using BVLC CaffeNet for image classification
model_dir = sprintf('../../../caffe/models/%s/', model);

prototxt_file = dir( fullfile(model_dir,'*deploy.prototxt') );   %# list all *.prototxt files
caffemodel_file = dir( fullfile(model_dir,'*.caffemodel') );   %# list all *.caffemodel files

net_model = [model_dir prototxt_file.name]; %Complete path to the protoxt file found in the directory
net_weights = [model_dir caffemodel_file.name]; %Complete path to the protoxt file found in the directory

phase = 'test'; % run with phase test (so that dropout isn't applied)

% Checking that the variable net_weights contains a file
if ~exist(net_weights, 'file')
  sMessage = sprintf('Please download %s from Model Zoo before you run this demo',model);
  error(sMessage);
end

% Initialise a network
fprintf('\n');
cprintf('*text', 'Initialising caffe network...\n'); 
net = caffe.Net(net_model, net_weights, phase);

% Some models require the dimensions of the images to be 224x224 while others require 227x227 
if isequal(model,'bvlc_alexnet')==1 || isequal(model,'bvlc_reference_caffenet')==1 ||isequal(model,'bvlc_reference_rcnn_ilsvrc13')==1 ||isequal(model,'placesCNN')==1
    net.blobs('data').reshape([227 227 3 1]);
else
    net.blobs('data').reshape([224 224 3 1]);
end


%% (2) Set paths & calculate number of images inside folders

pos_train_dir= sprintf('../%s/POS_TRAIN/%s/', dataset, category);
pos_num_train_images=length(dir(pos_train_dir))-2; % Unless you want to consider '.' and '..' as directories, you're probably going to want to subtract 2 from pos_num_train_images.


neg_train_dir= sprintf('../%s/NEG_TRAIN/%s/', dataset, category);
neg_num_train_images=length(dir(neg_train_dir))-2; % Unless you want to consider '.' and '..' as directories, you're probably going to want to subtract 2 from neg_num_train_images.


pos_test_dir= sprintf('../%s/POS_TEST/%s/', dataset,category);
pos_num_test_images=length(dir(pos_test_dir))-2; % Unless you want to consider '.' and '..' as directories, you're probably going to want to subtract 2 from pos_num_test_images.

neg_test_dir= sprintf('../%s/NEG_TEST/%s/', dataset,category);
neg_num_test_images=length(dir(neg_test_dir))-2; % Unless you want to consider '.' and '..' as directories, you're probably going to want to subtract 2 from neg_num_test_images.

% Checking if DS_STORE file exists in the path
if exist(sprintf('../%s/POS_TRAIN/%s/.DS_STORE', dataset, category), 'file') == 2
    pos_num_train_images = pos_num_train_images - 1;
end
if exist(sprintf('../%s/NEG_TRAIN/%s/.DS_STORE', dataset, category), 'file') == 2
    neg_num_train_images = neg_num_train_images - 1;
end
if exist(sprintf('../%s/POS_TEST/%s/.DS_STORE', dataset, category), 'file') == 2
    pos_num_test_images = pos_num_test_images - 1;
end
if exist(sprintf('../%s/NEG_TEST/%s/.DS_STORE', dataset, category), 'file') == 2
    neg_num_test_images = neg_num_test_images - 1;
end


%%NEW
pos_num_test_images = pos_num_test_images + initial_test_img_num;
neg_num_test_images = neg_num_test_images + initial_test_img_num;


tStart = tic;
%% (3) Training

% Positive Training
disp('Starting Positive Training')
h = waitbar(0,'Positive Training...','Name', 'TRAINING IN PROGRES');
for i = 1:pos_num_train_images
    sample_image_path =strcat(pos_train_dir,sprintf('%04d.jpg',i)); % Concatenate strings horizontally
    im = imread(sample_image_path);
    im = standardizeImage(im); % ensure of type single, w. three channels
    if isequal(model,'placesCNN')==1 
        mean_flag=1;
    else
        mean_flag=0;
    end
    
    % Prepare the image for the models requiring 224x224
    if isequal(model,'bvlc_googlenet')==1 || isequal(model,'ResNet50')==1 ||isequal(model,'ResNet101')==1 ||isequal(model,'ResNet152')==1 ||isequal(model,'VGG_CNN_M')==1 ||isequal(model,'VGG_CNN_S')==1||isequal(model,'VGG_CNN_F')==1||isequal(model,'VGGNet16')==1 ||isequal(model,'VGGNet19')==1
    	input_data = {prepare_image_224_224(im,mean_flag)};
        temp=net.forward(input_data);
        
        % Check if ResNet is selected so that pool5 layer is called instead of the default fc7 for the other models or inception_5b/output for bvlc_googlenet
        if isequal(model,'ResNet50')==1 || isequal(model,'ResNet101') || isequal(model,'ResNet152')
            data = net.blobs('res5c_branch2c').get_data;
            data = data(2:2:6, 2:2:6, : );
            code(:,i) = data( : );
            
        elseif isequal(model,'VGG_CNN_M')==1 || isequal(model,'VGGNet16')==1 || isequal(model,'VGGNet19')==1 || isequal(model,'VGG_CNN_S')==1 || isequal(model,'VGG_CNN_F')==1
            code(:,i) = net.blobs('fc7').get_data();
            
        elseif isequal(model,'bvlc_googlenet')==1 
            data = net.blobs('inception_5b/output').get_data;
            data = data(2:2:6, 2:2:6, : );
            code(:,i) = data( : );
            
        end
        
    else
    	input_data = {prepare_image(im,mean_flag)};
        temp=net.forward(input_data);
        code(:,i) = net.blobs('fc7').get_data();  
        
    end
    

    per = i / pos_num_train_images * 100;
    waitbar(i / pos_num_train_images,h,sprintf('Positive Training...%.2f%%',per));
    
end
close(h)
delete(h)
fprintf('\n');
cprintf('*text', '??? Positive Training \n');

% Negative Training 
h = waitbar(0,'Negative Training...','Name', 'TRAINING IN PROGRES');
for i = 1:neg_num_train_images
    j = i + pos_num_train_images;
    sample_image_path =strcat(neg_train_dir,sprintf('%04d.jpg',i)); % Concatenate strings horizontally
    im = imread(sample_image_path);
    im = standardizeImage(im); % ensure of type single, w. three channels
    if isequal(model,'placesCNN')==1 
        mean_flag=1;
    else
        mean_flag=0;
    end
   
    % Prepare the image for the models requiring 224x224
    if isequal(model,'bvlc_googlenet')==1 || isequal(model,'ResNet50')==1 ||isequal(model,'ResNet101')==1 ||isequal(model,'ResNet152')==1 ||isequal(model,'VGG_CNN_M')==1 ||isequal(model,'VGG_CNN_S')==1||isequal(model,'VGG_CNN_F')==1||isequal(model,'VGGNet16')==1 ||isequal(model,'VGGNet19')==1
    	input_data = {prepare_image_224_224(im,mean_flag)};
        temp=net.forward(input_data);
        
        % Check if ResNet is selected so that pool5 layer is called instead of the default fc7 for the other models or inception_5b/output for bvlc_googlenet
        if isequal(model,'ResNet50')==1 || isequal(model,'ResNet101') || isequal(model,'ResNet152')
            data = net.blobs('res5c_branch2c').get_data;
            data = data(2:2:6, 2:2:6, : );
            code_neg(:,j) = data( : );

        elseif isequal(model,'VGG_CNN_M')==1 || isequal(model,'VGGNet16')==1 || isequal(model,'VGGNet19')==1 || isequal(model,'VGG_CNN_S')==1 || isequal(model,'VGG_CNN_F')==1
            code_neg(:,j) = net.blobs('fc7').get_data();
            
        elseif isequal(model,'bvlc_googlenet')==1 
            data = net.blobs('inception_5b/output').get_data;
            data = data(2:2:6, 2:2:6, : );
            code_neg(:,j) = data( : );
            
        end
        
    else
    	input_data = {prepare_image(im,mean_flag)};
        temp=net.forward(input_data);
        code_neg(:,j) = net.blobs('fc7').get_data();
        
    end
    
    per = (i) / neg_num_train_images * 100;
    waitbar(i / neg_num_train_images,h,sprintf('Negative Training...%.2f%%',per));
    
end
close(h)
delete(h)
fprintf('\n');
cprintf('*text', '??? Negative Training \n');

%% (4) Testing 

% Positive testing 
h = waitbar(0,'Positive Testing...','Name', 'TESTING IN PROGRES');
disp(pos_num_test_images)

for i = 771:pos_num_test_images
    sample_image_path=strcat(pos_test_dir,sprintf('%04d.jpg',i));
    im = imread(sample_image_path);
    im = standardizeImage(im); % ensure of type single, w. three channels
    
    if isequal(model,'placesCNN')==1 
        mean_flag=1;
    else
        mean_flag=0;
    end
    
    % Prepare the image for the models requiring 224x224
    if isequal(model,'bvlc_googlenet')==1 || isequal(model,'ResNet50')==1 ||isequal(model,'ResNet101')==1 ||isequal(model,'ResNet152')==1 ||isequal(model,'VGG_CNN_M')==1 ||isequal(model,'VGG_CNN_S')==1||isequal(model,'VGG_CNN_F')==1||isequal(model,'VGGNet16')==1 ||isequal(model,'VGGNet19')==1
    	input_data = {prepare_image_224_224(im,mean_flag)};
        temp=net.forward(input_data);
        
        % Check if ResNet is selected so that pool5 layer is called instead of the default fc7 for the other models or inception_5b/output for bvlc_googlenet
        if isequal(model,'ResNet50')==1 || isequal(model,'ResNet101') || isequal(model,'ResNet152')
            data = net.blobs('res5c_branch2c').get_data;
            data = data(2:2:6, 2:2:6, : );
            code_v(:,i) = data( : );
            
        elseif isequal(model,'VGG_CNN_M')==1 || isequal(model,'VGGNet16')==1 || isequal(model,'VGGNet19')==1 || isequal(model,'VGG_CNN_S')==1 || isequal(model,'VGG_CNN_F')==1
            code_v(:,i) = net.blobs('fc7').get_data();
            
        elseif isequal(model,'bvlc_googlenet')==1 
            data = net.blobs('inception_5b/output').get_data;
            data = data(2:2:6, 2:2:6, : );
            code_v(:,i) = data( : );
            
        end
        
    else
    	input_data = {prepare_image(im,mean_flag)};
        temp=net.forward(input_data);
        code_v(:,i) = net.blobs('fc7').get_data();
        
    end

    per = i / pos_num_test_images * 100;
    waitbar(i / pos_num_test_images,h,sprintf('Positive Testing...%.2f%%',per));
      
end

close(h)
delete(h)
fprintf('\n');
cprintf('*text', '??? Positive Testing \n');

% Negative testing (Random images of animals)
h = waitbar(0,'Negative Testing...','Name', 'TESTING IN PROGRES');

for i = 771:neg_num_test_images
    j = i + pos_num_test_images;
    sample_image_path =strcat(neg_test_dir,sprintf('%04d.jpg',i)); % Concatenate strings horizontally
    im = imread(sample_image_path);
    im = standardizeImage(im); % ensure of type single, w. three channels
    
    if isequal(model,'placesCNN')==1 
        mean_flag=1;
    else
        mean_flag=0;
    end
    
    % Prepare the image for the models requiring 224x224
    if isequal(model,'bvlc_googlenet')==1 || isequal(model,'ResNet50')==1 ||isequal(model,'ResNet101')==1 ||isequal(model,'ResNet152')==1 ||isequal(model,'VGG_CNN_M')==1 ||isequal(model,'VGG_CNN_S')==1||isequal(model,'VGG_CNN_F')==1||isequal(model,'VGGNet16')==1 ||isequal(model,'VGGNet19')==1
    	input_data = {prepare_image_224_224(im,mean_flag)};
        temp=net.forward(input_data);
        
        % Check if ResNet is selected so that pool5 layer is called instead of the default fc7 for the other models or inception_5b/output for bvlc_googlenet
        if isequal(model,'ResNet50')==1 || isequal(model,'ResNet101') || isequal(model,'ResNet152')
            data = net.blobs('res5c_branch2c').get_data;
            data = data(2:2:6, 2:2:6, : );
            code_v_neg(:,j) = data( : );
            
        elseif isequal(model,'VGG_CNN_M')==1 || isequal(model,'VGGNet16')==1 || isequal(model,'VGGNet19')==1 || isequal(model,'VGG_CNN_S')==1 || isequal(model,'VGG_CNN_F')==1
            code_v_neg(:,j) = net.blobs('fc7').get_data();
            
        elseif isequal(model,'bvlc_googlenet')==1 
            data = net.blobs('inception_5b/output').get_data;
            data = data(2:2:6, 2:2:6, : );
            code_v_neg(:,j) = data( : );
            
        end
        
    else
    	input_data = {prepare_image(im,mean_flag)};
        temp=net.forward(input_data);
        code_v_neg(:,j) = net.blobs('fc7').get_data();
        
    end
    
    per = (i) / neg_num_test_images * 100;
    waitbar(i / neg_num_test_images,h,sprintf('Negative Testing...%.2f%%',per));
    
end
close(h)
delete(h)

fprintf('\n');
cprintf('*text', '??? Negative Testing \n');
fprintf('\n');
fprintf('\n');

% Getting only the features with non zero values in the negative results
code_v = code_v(:,771:1100);
code_v_neg = code_v_neg(:,1871:2200);
code_neg = code_neg(:,771:1540);


%% Time elapsed for the whole process 
fprintf('\n');
tEnd = toc(tStart);
  
hours = floor(tEnd / 3600);

tEnd = tEnd - hours * 3600;
mins = floor(tEnd / 60);
secs = tEnd - mins * 60;

fprintf('Total Time Elapsed:  %.d hours & %.d minutes\n',floor(hours),mins);

%% Functions for preparing images for caffe
function cropped_data = prepare_image(im,mean_flag)
if mean_flag==1
    d = load('../../../caffe/models/placesCNN/places_mean.mat');
    mean_data = d.image_mean;
else
    d = load('../../../caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
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


function cropped_data = prepare_image_224_224(im,mean_flag)
if mean_flag==1
    d = load('../../../caffe/models/placesCNN/places_mean.mat');
    mean_data = d.image_mean;
else
    d = load('../../../caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');
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






