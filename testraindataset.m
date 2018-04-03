% weatherclassification
%       Function that trains a series of classifiers using different models
%       as feature extractors

% Parameters
%   models - array of strings specifying the models to use
%   categories - array of the categories to be used for the current
%       iteration
%   superpixels - number of superpixels to use (0 if no superpixels are to
%       be used
%   C - constant for the classifier
%   runid - char sequence to identify a given execution of the script


function testraindataset()

    % Adding relevant paths


    % Indicate that the predefined structure of files will be used
    predefined = true;

    % By default, raw images will be used unless specified with superpixels var
    images_type = 'normal';

    % Define parameters
    % The following lines have the all the models and categories used for the
    % work
    categories = {sprintf('Cloudy'), sprintf('Foggy'), sprintf('Rainy'), sprintf('Snowy'), sprintf('Sunny')};
    models = {sprintf('bvlc_googlenet'),sprintf('placesCNN'),sprintf('ResNet50'),sprintf('ResNet101'),sprintf('ResNet152'),sprintf('VGG_CNN_F'),sprintf('VGG_CNN_M'),sprintf('VGG_CNN_S'),sprintf('VGGNet16'),sprintf('VGGNet19') };
    NoOfCategories = size(categories);
    numsuperpixels = [25 50 75 100];
    color = {sprintf('black'), sprintf('blue'), sprintf('green'), sprintf('yellow'), sprintf('white'), sprintf('cyan'), sprintf('magenta'),sprintf('red')};
    NoOfColor = size(color);
    C = 10;
    skip = false;

    
        % The ExtendedWeatherDatabase with the required format must be built
        
         database = 'ExtendedWeatherDatabase_SP';
            % creation of folders : required if initializing
             ewd_spbasedir1 = sprintf('/home/zeba/%s',database);
%             
%               for i = 1: NoOfColor(2)
%                 ewd_spbasedir2 = sprintf('%s/%s',ewd_spbasedir1,color{1,i});
%             for j = 1 : length(numsuperpixels)
%                  
%                 ewd_spbasedir4 = sprintf('%s/%d',ewd_spbasedir2,numsuperpixels(j));
%                  mkdir(ewd_spbasedir4);
%                                  
%                  for k = 1:NoOfCategories(2)
%                     ewd_spbasedir = sprintf('%s/POS_TRAIN/%s',ewd_spbasedir4,categories{1,k});
%                     mkdir(ewd_spbasedir);
%                     ewd_spbasedir = sprintf('%s/POS_TEST/%s',ewd_spbasedir4,categories{1,k});
%                     mkdir(ewd_spbasedir)
%                     ewd_spbasedir = sprintf('%s/NEG_TRAIN/%s',ewd_spbasedir4,categories{1,k});
%                     mkdir(ewd_spbasedir)
%                     ewd_spbasedir = sprintf('%s/NEG_TEST/%s',ewd_spbasedir4,categories{1,k});
%                     mkdir(ewd_spbasedir)
%                  end
%             end
%             end
            
for  z = 1:NoOfColor(2) 
               ewd_spbasedir2 = sprintf('%s/%s',ewd_spbasedir1,color{1,z});
   for y = 1:length(numsuperpixels)
                    ewd_spbasedir = sprintf('%s/%d/',ewd_spbasedir2,numsuperpixels(y))
            disp('[LOG] Preparing Dataset with Positive and Negative Partitions')
            % Build the datasets as the matdeeprep function requires them
            % Copy to POS_TRAIN subdirectory
            for i = 1:NoOfCategories(2)
               for j = 1:770
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,i}, sprintf('%04d',j)), [ewd_spbasedir sprintf('POS_TRAIN/%s/', categories{1,i})]);
               end
            end

            % Copy to POS_TEST subdirectory
            for i = 1:NoOfCategories(2)
               for j = 771:1100
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,i}, sprintf('%04d',j)), [ewd_spbasedir sprintf('POS_TEST/%s/', categories{1,i})]);
               end
            end

            % Copy to NEG_TRAIN subdirectory
            %   A manual selection of specific images is made to achieve a
            %   negative set that involves all of the different categories
            for i = 1:NoOfCategories(2)
               %Take images 1-193
               k = (1 + mod(i+1-1, NoOfCategories(2)));
               for j = 1:193
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,k}, sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TRAIN/%s/', categories{1,i})]);
               end

               %Take images 194-386
               k = (1 + mod(i+2-1, NoOfCategories(2)));
               for j = 194:386
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,k}, sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TRAIN/%s/', categories{1,i})]);
               end

               %Take images 387-578
               k = (1 + mod(i+3-1, NoOfCategories(2)));
               for j = 387:578
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,k}, sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TRAIN/%s/', categories{1,i})]);
               end

               %Take images 579-770
               k = (1 + mod(i+4-1,NoOfCategories(2) ));
               for j = 579:770
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,k}, sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TRAIN/%s/', categories{1,i})]);
               end
            end

            %Copy to NEG_TEST subdirectory
            %   A manual selection of specific images is made to achieve a
            %   negative set that involves all of the different categories
            for i = 1:NoOfCategories(2)
               %Take images 771-853
               k = (1 + mod(i+1-1, NoOfCategories(2)));
               for j = 771:853
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,k}, sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TEST/%s/', categories{1,i})]);
               end

               %Take images 854-936
               k = (1 + mod(i+2-1, NoOfCategories(2)));
               for j = 854:936
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,k}, sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TEST/%s/', categories{1,i})]);
               end

               %Take images 937-1018
               k = (1 + mod(i+3-1, NoOfCategories(2)));
               for j = 937:1018
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,k}, sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TEST/%s/', categories{1,i})]);
               end

               %Take images 1019-1100
               k = (1 + mod(i+4-1, NoOfCategories(2)));
               for j = 1019:1100
                   copyfile(sprintf('/home/zeba/superPixel/%s/%d/%s/%s.jpg',color{1,z},numsuperpixels(y),categories{1,k}, sprintf('%04d',j)), [ewd_spbasedir sprintf('NEG_TEST/%s/', categories{1,i})]);
               end
            end
        end
    end
end
    
