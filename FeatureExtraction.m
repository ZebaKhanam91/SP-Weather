% Feature Extraction 
%       Function that trains a series of classifiers using different models
%       as feature extractors

% Parameters
%   models - array of strings specifying the models to use
%   categories - array of the categories to be used for the current
%       iteration
%   superpixels - number of superpixels to use (0 if no superpixels are to
%       be used
%   color - constant for the classifier
%   runid - char sequence to identify a given execution of the script


    % Adding relevant paths
    addpath('/home/zk17735/caffe');
    addpath('/home/zk17735/caffe/matlab/demo');
    addpath('/home/zk17735/caffe/models/spModels');
    addpath('/home/zk17735/caffe/matlab/demo/SP-Weather/ExternalCode');

    % Indicate that the predefined structure of files will be used
    predefined = true;

    % By default, raw images will be used unless specified with superpixels var
    images_type = 'normal';

    % Define parameters
    % The following lines have the all the models and categories used for the
    % work
    categories = {sprintf('Cloudy'), sprintf('Foggy'), sprintf('Rainy'), sprintf('Snowy'), sprintf('Sunny')};
    models = {sprintf('bvlc_reference_caffenet'),sprintf('placesCNN'),sprintf('ResNet50'),sprintf('ResNet101'),sprintf('ResNet152'),sprintf('VGG_CNN_F'),sprintf('VGG_CNN_M'),sprintf('VGG_CNN_S'),sprintf('VGGNet16'),sprintf('VGGNet19') };
    NoOfCategories = size(categories);
    NoOfModels = size(models);
    numsuperpixels = [25];
    color = {sprintf('blue')};
    NoOfColor = size(color);
    C = 10;
    
    %models = [string('bvlc_googlenet'),string('placesCNN'),string('ResNet50'),string('ResNet101'),string('ResNet152'),string('VGG_CNN_F'),string('VGG_CNN_M'),string('VGG_CNN_S'),string('VGGNet16'),string('VGGNet19') ];
    %categories = [string('cloudy'), string('foggy'), string('rainy'), string('snowy'), string('sunny')];
    %models = {sprintf('GaussNet'),sprintf('finetune_flickr_style'),sprintf('bvlc_alexnet'),sprintf('blvc_googlenet'),sprintf('SqueezeNet'),sprintf('bvlc_reference_rcnn_ilsvrc13'),sprintf('bvlc_reference_caffenet'),sprintf('placesCNN'),sprintf('ResNet50'),sprintf('ResNet101'),sprintf('ResNet152'),sprintf('VGG_CNN_F'),sprintf('VGG_CNN_M'),sprintf('VGG_CNN_S'),sprintf('VGGNet16'),sprintf('VGGNet19') };
    % Change the following line to skip all steps of the function except the
    % classification part
   
    

    % Extract features with selected model
    % Directory for extracted features -> ../extractedFeatures/
    %mkdir('../extractedFeatures');
    features_base_dir = '/home/zk17735/caffe/matlab/extractedFeatures';
for s = 1:NoOfColor(2)
    
    feature_COL_dir = sprintf('%s/%s',features_base_dir,color{1,s});
    mkdir(feature_COL_dir);
    
   for w = 1: length(numsuperpixels)
    
       feature_SP_dir = sprintf('%s/%d',feature_COL_dir,numsuperpixels(w));
       mkdir(feature_SP_dir);
    % Check if Superpixels are being used in order of pointing to the correct
    % directory
    
        disp('[LOG] Extracting features (with superpixels)')

       % If there are superpixels, the images are in drectoriesforspimages directories 
       for i=1:NoOfModels(2)
           
           feature_Model_dir = sprintf('%s/%s',feature_SP_dir,models{1,i});
           mkdir(feature_Model_dir);
           
           for j = 1:NoOfCategories(2)

               % Point to the corresponding directory
               current_dir = sprintf('%s/%s',feature_Model_dir,categories{1,j});
               mkdir(current_dir)
               
               % Call the matdeeprep function and retrieve its output values
               [code, code_v, code_neg, code_v_neg] = matdeeprep(models{1,i}, sprintf('/home/zk17735/SP/Dataset/ExtendedWeatherDatabase_SP/%s/%d',color{1,s},numsuperpixels(w)),categories{1,j});

               % Save the features for future use (This allows for the use of
               % the extracted features in other settings
               save(sprintf('%s/positive_train_features.mat',current_dir), 'code');
               save(sprintf('%s/positive_test_features.mat',current_dir), 'code_v');
               save(sprintf('%s/negative_train_features.mat',current_dir), 'code_neg');
               save(sprintf('%s/negative_test_features.mat',current_dir), 'code_v_neg');

               % Clean the variables that contain the output from matdeeprep
               code = [];
               code_v = [];
               code_neg = [];
               code_v_neg = [];
           end
       end
   end
end

   
    

