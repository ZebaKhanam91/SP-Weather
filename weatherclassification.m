% weatherclassification
%       Function that trains a series of classifiers using different models
%       as feature extractors

% Parameters
%   models - array of chars specifying the models to use
%   categories - array of the categories to be used for the current
%       iteration
%   superpixels - number of superpixels to use (0 if no superpixels are to
%       be used
%   C - constant for the classifier
%   runid - char sequence to identify a given execution of the script


function weatherclassification()

    % Adding relevant paths
    addpath('/home/zeba/caffe/matlab/demo/');
    addpath('/home/zeba/caffe/models/');

    % Indicate that the predefined structure of files will be used
    predefined = true;
   
    % By default, raw images will be used unless specified with superpixels var
    images_type = 'normal';

    % Define parameters
    % The following lines have the all the models and categories used for the
    % work
    categories = {sprintf('Cloudy'), sprintf('Foggy'), sprintf('Rainy'), sprintf('Snowy'), sprintf('Sunny')};
    models = {sprintf('bvlc_googlenet'),sprintf('placesCNN'),sprintf('ResNet50'),sprintf('ResNet101'),sprintf('ResNet152'),sprintf('VGG_CNN_F'),sprintf('VGG_CNN_M'),sprintf('VGG_CNN_S'),sprintf('VGGNet16'),sprintf('VGGNet19') };
    NoOfCategories = size(categories)
    numsuperpixels = 25
    C = 10
    runid = 'a'
    % Change the following line to skip all steps of the function except the
    % classification part
    skip = false;

    % Change the following line if the images are already marked with
    % superpixels (No need to use execution time in that step)
    skipsuperpixels = false;

    if ~skipsuperpixels
    %Identify images (without superpixels)
    %   Find the corresponding directory


    % Mark images with superpixels
    %   use the script called ProcessPictures.m

    % Arrays to store the required directory paths
    origindirectories = {};
    directoriesforspimages = {};
    DatasetPath = '/home/zeba/Code/Dataset/';
    MarkedDatasetPath = '/home/zeba/Dataset/marked/';

    disp('[LOG] Gathering relevant directories')

    % For the specified categories, build the corresponding path
    for i = 1:1:NoOfCategories(2)
       directoryoriginaux = [DatasetPath char(categories(i))];
       origindirectories{end + 1} = directoryoriginaux;
       directoryfinalaux = [MarkedDatasetPath char(categories(i))];
       directoriesforspimages{end+1} = directoryfinalaux ;
     
    end

    % If the number of superpixels is above 0, then we most process the images
    if numsuperpixels > 0
        %Change the type of image to be used
        images_type = 'superpixel';

        disp('[LOG] Marking Superpixels')

        % Specify the dataset to be used
        database = 'ExtendedWeatherDatabase_SP';

        % For all the categories, process each of the contained images with the
        % custom function to add the superpixel mask
        
        for i = 1:NoOfCategories(2)
            display1 = sprintf('[LOG] Starting Category %s', categories{1,i});
            disp(display1)
            processpictures(origindirectories{1,i}, directoriesforspimages{1,i}, numsuperpixels)
        end

%         % The ExtendedWeatherDatabase with the required format must be built
%         if predefined
% 
%             % Load the images from the 'marked' folders
%             ewd_spbasedir = sprintf('../%s/', database);
% 
%             disp('[LOG] Preparing Dataset with Positive and Negative Partitions')
%             % Build the datasets as the matdeeprep function requires them
%             % Copy to POS_TRAIN subdirectory
%             for i = 1:numel(categories)
%                for j = 1:770
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(i), sprintf('%04d',j)), [ewd_spbasedir sprintf('POS_TRAIN/%s/', categories(i))]);
%                end
%             end
% 
%             % Copy to POS_TEST subdirectory
%             for i = 1:numel(categories)
%                for j = 771:1100
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(i), sprintf('%04d',j)), [ewd_spbasedir sprintf('POS_TEST/%s/', categories(i))]);
%                end
%             end
% 
%             % Copy to NEG_TRAIN subdirectory
%             %   A manual selection of specific images is made to achieve a
%             %   negative set that involves all of the different categories
%             for i = 1:numel(categories)
%                %Take images 1-193
%                k = (1 + mod(i+1-1, numel(categories)));
%                for j = 1:193
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(k), sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TRAIN/%s/', categories(i))]);
%                end
% 
%                %Take images 194-386
%                k = (1 + mod(i+2-1, numel(categories)));
%                for j = 194:386
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(k), sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TRAIN/%s/', categories(i))]);
%                end
% 
%                %Take images 387-578
%                k = (1 + mod(i+3-1, numel(categories)));
%                for j = 387:578
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(k), sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TRAIN/%s/', categories(i))]);
%                end
% 
%                %Take images 579-770
%                k = (1 + mod(i+4-1, numel(categories)));
%                for j = 579:770
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(k), sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TRAIN/%s/', categories(i))]);
%                end
%             end
% 
%             %Copy to NEG_TEST subdirectory
%             %   A manual selection of specific images is made to achieve a
%             %   negative set that involves all of the different categories
%             for i = 1:numel(categories)
%                %Take images 771-853
%                k = (1 + mod(i+1-1, numel(categories)));
%                for j = 771:853
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(k), sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TEST/%s/', categories(i))]);
%                end
% 
%                %Take images 854-936
%                k = (1 + mod(i+2-1, numel(categories)));
%                for j = 854:936
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(k), sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TEST/%s/', categories(i))]);
%                end
% 
%                %Take images 937-1018
%                k = (1 + mod(i+3-1, numel(categories)));
%                for j = 937:1018
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(k), sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TEST/%s/', categories(i))]);
%                end
% 
%                %Take images 1019-1100
%                k = (1 + mod(i+4-1, numel(categories)));
%                for j = 1019:1100
%                    copyfile(sprintf('../Dataset/%s_marked/%s.jpg',categories(k), sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TEST/%s/', categories(i))]);
%                end
%             end
%         end
    end
    end
% 
%     % Extract features with selected model
%     % Directory for extracted features -> ../extractedFeatures/
%     features_base_dir = '../extractedFeatures/';
% 
%     if ~skip
%     % Check if Superpixels are being used in order of pointing to the correct
%     % directory
%     if numsuperpixels > 0
%         disp('[LOG] Extracting features (with superpixels)')
% 
%        % If there are superpixels, the images are in drectoriesforspimages directories 
%        for i=1:numel(models)
%            for j = 1:numel(categories)
% 
%                % Point to the corresponding directory
%                current_dir = char(features_base_dir) + char(sprintf('%s/%s/', models(i), categories(j)));
% 
%                % Call the matdeeprep function and retrieve its output values
%                [code, code_v, code_neg, code_v_neg] = matdeeprep(char(models(i)), 'ExtendedWeatherDatabase_SP', char(categories(j)));
% 
%                % Save the features for future use (This allows for the use of
%                % the extracted features in other settings
%                save(char(current_dir + char('positive_train_features.mat')), 'code');
%                save(char(current_dir + char('positive_test_features.mat')), 'code_v');
%                save(char(current_dir + char('negative_train_features.mat')), 'code_neg');
%                save(char(current_dir + char('negative_test_features.mat')), 'code_v_neg');
% 
%                % Clean the variables that contain the output from matdeeprep
%                code = [];
%                code_v = [];
%                code_neg = [];
%                code_v_neg = [];
%            end
%        end
%     else
%         % If there are no superpixels, the images are in the origindirectories
%         % directories
%         disp('[LOG] Extracting features (without superpixels)')
%         for i=1:numel(models)
%            for j = 1:numel(categories)
%                 % Point to the corresponding directory
%                 current_dir = char(features_base_dir) + char(sprintf('%s/%s/', models(i), categories(j)));
% 
%                 % Call the matdeeprep function and retrieve its output values
%                 [code, code_v, code_neg, code_v_neg] = matdeeprep(char(models(i)), 'ExtendedWeatherDatabase', char(categories(j)));
% 
%                 % Save the features for future use (This allows for the use of
%                 % the extracted features in other settings
%                 save(char(current_dir + char('positive_train_features.mat')), 'code');
%                 save(char(current_dir + char('positive_test_features.mat')), 'code_v');
%                 save(char(current_dir + char('negative_train_features.mat')), 'code_neg');
%                 save(char(current_dir + char('negative_test_features.mat')), 'code_v_neg');
% 
%                 % Clean the variables that contain the output from matdeeprep
%                 code = [];
%                 code_v = [];
%                 code_neg = [];
%                 code_v_neg = [];
%            end
%        end
%     end
%     end
% 
%     %Train and Test Classifier with extracted features
% 
%     disp('[LOG] Training and Testing Process')
% 
%     % Make a directory to store intermediate results of the current execution
%     mkdir(sprintf('../IntermediateResults/%s/', runid));
% 
%     % For each model 
%     for i = (1:1:length(models))
%         disp(fprintf('[LOG] Model %s \n', models(i)))
% 
%         % For each category
%         for j = (1:1:length(categories))
%             disp(fprintf('[LOG] Category %s \n', categories(j)))
% 
% 
%             base_dir = sprintf('../IntermediateResults/%s/%s/%s/', runid, models(i), categories(j));
%             mkdir(base_dir);
%             % Call the WeatherClassifier_TwoClass function and retrieve its output values
%             % Train the classifier, getting is results as output
%             % Average Precision is in the info.auc struct
%             [train_w,train_bias,train_scores,test_scores,info]=WeatherClassifier_TwoClass(models(i), categories(j), images_type, C);
% 
%             disp(sprintf('Model %s for category %s has AP of %s', models(i), categories(j), info.auc))
% 
%             % Save the output from the classifier in a MATLAB file
%             save([base_dir 'results.mat'],'train_w','train_bias','train_scores','test_scores','info');
% 
%             % Clear the variables holding the classifier results
%             clear train_w train_bias test_w test_bias train_scores test_scores info;
% 
%       % Save results to a file
%         end
%     end
% 
%     %Write results to CSV file
%     %   This script generates a CSV files with the results of the experiment
%     %   found in the results folders in the directroy of the project
%     %   After the execution of this file, a CSV file is created with name
%     %   WeatherClassification_Results.csv 
%     %   In the CSV file, each column is a different model used for the feature
%     %   extraction and each line is one of the different categories
%     %   The columns have the models in the following order:
%     %   'bvlc_googlenet','placesCNN','ResNet50','ResNet101','ResNet152','VGG_CNN_F','VGG_CNN_M','VGG_CNN_S','VGGNet16','VGGNet19'
%     %   The rows have the categories in the following order:
%     %   'cloudy' (first row),'foggy'(second row),'rainy' (third row),'snowy','sunny'
%     %   The results show the average precision for each case
% 
%     results = sprintf('../IntermediateResults/%s/', runid);
% 
%     disp('[LOG] Preparing Results')
% 
%     complete_values = []; % Variable to store the complete set of results
%     ap_values = []; % Variable to store the results of each row
% 
%     % Load results
%     % For each category
%     for i = 1:1:size(categories,2)
%         % For each model
%         for j = 1:1:size(models,2)
%             % Load the results generated by the different models for each
%             % category
%             res = load([results sprintf('%s/%s/results.mat',models(j), categories(i))]);
% 
%             % Retrieve the AP value for each category and store it in an array
%             ap_values = [ap_values res.info.ap];
%         end
% 
%         % Collect all the results into a single matrix
%         complete_values = vertcat(complete_values, ap_values);
%         ap_values = [];
%     end
%     
%     % Saving the results to the specified folder using the current
%     % execution id
%     finalresultsdir = sprintf('../FinalResults/%s/', runid);
%     mkdir(finalresultsdir);
%     finalresultsdir = [finalresultsdir 'WeatherClassResults.csv'];
%     csvwrite(finalresultsdir, complete_values);
%     disp(sprintf('[LOG] Results available in %s', finalresultsdir))
% end
