%% weatherclassification_validate
%   This function trains classifiers a number of times and returns the
%   average precision for each category in a matrix as well as the
%   collection of trained classifiers.

% Parameters
%   initialdirectory - Directory where the features to be used are stored
%                       (as a MATLAB file)
%   model - Model to be used in a given execution
%   numberofruns - Number of iterations expected to be carried out
%   C - Regularisation parameter for the SVM

% Output:
%   The format of the results matrix is each category in each row and one
%   run per column.
%   The cell array called trainedclassifiers is a cell array where each
%   element of the array is a cell array with the train weights, the train
%   bias and the information from the testing phase.

function [results, trainedclassifiers] = weatherclassification_validate(initialdirectory, model, numberofruns, C)
    %initialdirectory = '../extractedFeatures/';
    %model = 'bvlc';

    % Getting the data extracted by the chosen model

    disp('[LOG] Getting features from files \n')

    % Auxiliary variables
    models = [string('bvlc_googlenet'),string('placesCNN'),string('ResNet50'),string('ResNet101'),string('ResNet152'),string('VGG_CNN_F'),string('VGG_CNN_M'),string('VGG_CNN_S'),string('VGGNet16'),string('VGGNet19') ];
    number_features = [9216 4096 18432 18432 18432 4096 4096 4096 4096 4096];
    model_to_use = strfind(models, model);
    curr_model = 0;
    model = char(model);

    % Finding the current model to assign the correct number that represents
    % the number of features corresponding to the model
    for i = 1:1:size(model_to_use,2)
        if ~isempty(model_to_use{i})
        curr_model = i;
        end
    end
    numberfeatures = number_features(curr_model);

    %   Retrieving the features previously extracted and stored in known
    %   location

    cloudy_train = load([initialdirectory model '/cloudy/positive_train_features.mat']);
    cloudy_train = cloudy_train.code;
    cloudy_test = load([initialdirectory model '/cloudy/positive_test_features.mat']);
    cloudy_test = cloudy_test.code_v;

    foggy_train = load([initialdirectory model '/foggy/positive_train_features.mat']);
    foggy_train = foggy_train.code;
    foggy_test = load([initialdirectory model '/foggy/positive_test_features.mat']);
    foggy_test = foggy_test.code_v;

    rainy_train = load([initialdirectory model '/rainy/positive_train_features.mat']);
    rainy_train = rainy_train.code;
    rainy_test = load([initialdirectory model '/rainy/positive_test_features.mat']);
    rainy_test = rainy_test.code_v;

    snowy_train = load([initialdirectory model '/snowy/positive_train_features.mat']);
    snowy_train = snowy_train.code;
    snowy_test = load([initialdirectory model '/snowy/positive_test_features.mat']);
    snowy_test = snowy_test.code_v;

    sunny_train = load([initialdirectory model '/sunny/positive_train_features.mat']);
    sunny_train = sunny_train.code;
    sunny_test = load([initialdirectory model '/sunny/positive_test_features.mat']);
    sunny_test = sunny_test.code_v;


    %Joining all the data

    disp('[LOG] Joining training and testing data \n')

    cloudy_data = [cloudy_train cloudy_test];
    foggy_data = [foggy_train foggy_test];
    rainy_data = [rainy_train rainy_test];
    snowy_data = [snowy_train snowy_test];
    sunny_data = [sunny_train sunny_test];
    
    % Auxiliary variables
    results = [];
    result_aux = [];
    trainedclassifiers = {};
    trainedclassifier_aux = {};
    trainedclassifiers_temp = {};

    % Once everythnig is put together, perform multiple classifications 
    disp(sprintf('[LOG] Starting validations for model %s \n', model))
    for i = 1:numberofruns

        disp(sprintf('[LOG] Run number %s \n', string(i)))

        % Shuffle all the categories
        cloudy_data = cloudy_data(:, randperm(size(cloudy_data,2)));
        foggy_data = foggy_data(:, randperm(size(foggy_data,2)));
        rainy_data = rainy_data(:, randperm(size(rainy_data,2)));
        snowy_data = snowy_data(:, randperm(size(snowy_data,2)));
        sunny_data = sunny_data(:, randperm(size(sunny_data,2)));

        categories = {cloudy_data, foggy_data, rainy_data, snowy_data, sunny_data};
        
        % Adding the labels
        labels_train = ones(770,1)';
        labels_train_neg = -1 * ones(770,1)';

        labels_test = ones(330,1)';
        labels_test_neg = -1 * ones(330,1)';


        for j = 1:size(categories,2)

            disp(sprintf('[LOG] Processing category %s \n', string(j)))

            positive_training = categories{j}(:, 1:770);
            positive_testing = categories{j}(:, 771:1100);
            negative_training = [];
            negative_testing = [];


            % Negative Training

               %Take images 1-193
               k = (1 + mod(j+1-1, size(categories,2)));
               negative_training = [negative_training categories{k}(:,1:193)];

               %Take images 194-386
               k = (1 + mod(j+2-1, size(categories,2)));
               negative_training = [negative_training categories{k}(:,194:386)];

               %Take images 387-578
               k = (1 + mod(j+3-1, size(categories,2)));
               negative_training = [negative_training categories{k}(:,387:578)];

               %Take images 579-770
               k = (1 + mod(j+4-1, size(categories,2)));
               negative_training = [negative_training categories{k}(:,579:770)];


            % Negative Testing
               %Take images 771-853
               k = (1 + mod(j+1-1, size(categories,2)));
               negative_testing = [negative_testing categories{k}(:,771:853)];

               %Take images 854-936
               k = (1 + mod(j+2-1, size(categories,2)));
               negative_testing = [negative_testing categories{k}(:,854:936)];

               %Take images 937-1018
               k = (1 + mod(j+3-1, size(categories,2)));
               negative_testing = [negative_testing categories{k}(:,937:1018)];

               %Take images 1019-1100
               k = (1 + mod(j+4-1, size(categories,2)));
               negative_testing = [negative_testing categories{k}(:,1019:1100)];

               % Mix Positive and Negative Data

               positive_training = vertcat(positive_training, labels_train);
               positive_testing = vertcat(positive_testing, labels_test);
               negative_training = vertcat(negative_training, labels_train_neg);
               negative_testing = vertcat(negative_testing, labels_test_neg);

               complete_data_train = [positive_training negative_training];
               complete_data_test = [positive_testing negative_testing];

               %Shuffling to have the positive and negative categories mixed
               complete_data_train = complete_data_train(:, randperm(size(complete_data_train,2)));
               complete_data_test = complete_data_test(:, randperm(size(complete_data_test,2)));

               %Separating the data and the labels
               data_to_train = complete_data_train(1:numberfeatures, :);
               labels_to_train = complete_data_train(numberfeatures + 1, :);

               data_to_test = complete_data_test(1:numberfeatures, :);
               labels_to_test = complete_data_test(numberfeatures + 1, :);

               %   The following lines of code are from the category recognition prctical
               %   source code with slight modifications
               %   (From next line of code until comment containing *******)
               %    Source:
               %    A. Vedaldi and A. Zisserman, ?Recognition of object categories practical,? 
               %    University of Oxford, 2015. [Online]. 
               %    Available: http://www.robots.ox.ac.uk/~vgg/practicals/category-recognition/index.html. 
               %    [Accessed 14 June 2017].

               % Training the classifier

               [train_w, train_bias] = trainLinearSVM(data_to_train, labels_to_train, C);

               % Getting Training Score
               train_scores = train_w' * data_to_train + train_bias;

               % Testing the classifier
               test_scores = train_w' * data_to_test + train_bias;

               [drop,drop,info] = vl_pr(labels_to_test, test_scores);

               % *******

               % Storing the results of the current iteration
               result_aux = vertcat(result_aux, info.auc);
               trainedclassifier_aux = {train_w, train_bias, info};
               trainedclassifiers_temp = [trainedclassifiers_temp trainedclassifier_aux];

               clear positive_training positive_testing negative_training negative_testing;

        end
        
        % Storing the results of the current iteration and preparing
        % variables for next iteration
        results = [results result_aux];
        trainedclassifiers = [trainedclassifiers trainedclassifiers_temp];
        result_aux = [];
        trainedclassifiers_temp = {};
    end
end
