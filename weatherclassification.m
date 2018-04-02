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

    % Define parameters
    % The following lines have the all the models and categories used for the
    % work
    categories = {sprintf('Cloudy'), sprintf('Foggy'), sprintf('Rainy'), sprintf('Snowy'), sprintf('Sunny')};
    color = ['k' 'b' 'g' 'r' 'm' 'y' 'w'];
    models = {sprintf('bvlc_googlenet'),sprintf('placesCNN'),sprintf('ResNet50'),sprintf('ResNet101'),sprintf('ResNet152'),sprintf('VGG_CNN_F'),sprintf('VGG_CNN_M'),sprintf('VGG_CNN_S'),sprintf('VGGNet16'),sprintf('VGGNet19') };
    NoOfCategories = size(categories);
    numsuperpixels = [25 50 75 100];
    

    % Change the following line if the images are already marked with
    % superpixels (No need to use execution time in that step)
    skipsuperpixels = false;
for s = 1 : length(color) 
    mkdir('C:\\Dataset',color(s));
  for j = 1:4
    if ~skipsuperpixels
    %Identify images (without superpixels)
    %   Find the corresponding directory


    % Mark images with superpixels
    %   use the script called ProcessPictures.m

    % Arrays to store the required directory paths
    origindirectories = {};
    directoriesforspimages = {};
    MarkedDatasetPath = sprintf('C:\\Dataset\\%c',color(s));
    mkdir(MarkedDatasetPath,char(string(numsuperpixels(j))));
    DatasetPath = 'C:\Dataset\';
    MarkedDatasetPath = sprintf('C:\\Dataset\\%c\\%d\\',color(s),numsuperpixels(j));
    

    disp('[LOG] Gathering relevant directories')

    % For the specified categories, build the corresponding path
    for i = 1:1:NoOfCategories(2)
       mkdir(MarkedDatasetPath,char(categories(i)));
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
        
        for i = 1:1:NoOfCategories(2)
            display1 = sprintf('[LOG] Starting Category %s', categories{1,i});
            disp(display1)
            processpictures(origindirectories{1,i}, directoriesforspimages{1,i}, numsuperpixels(j),color(s))
        end


    end
    end
  end
end
