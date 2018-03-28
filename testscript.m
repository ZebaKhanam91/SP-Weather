categories = {sprintf('Cloudy'), sprintf('Foggy'), sprintf('Rainy'), sprintf('Snowy'), sprintf('Sunny')};
models = {sprintf('bvlc_googlenet'),sprintf('placesCNN'),sprintf('ResNet50'),sprintf('ResNet101'),sprintf('ResNet152'),sprintf('VGG_CNN_F'),sprintf('VGG_CNN_M'),sprintf('VGG_CNN_S'),sprintf('VGGNet16'),sprintf('VGGNet19') };
origindirectories = {};
directoriesforspimages = {};
DatasetPath = '/home/zeba/Code/Dataset/Cloudy/*.jpg';
MarkedDatasetPath = '/home/zeba/Dataset/%s_marked/';
NoOfCategories = size(categories);
files = dir(DatasetPath);
 count = 1;
 file = files'
    for i = 1:length(files)
        if mod(count,10) == 0
                disp(sprintf('[LOG] Processed image %s', file.name))
        end
        %Identifying the name of the current image
        aux = file.name;

        %Reading the file with the current image
        A = imread(char(strcat(originDir,string(aux))));

        %Using the custom image that adds the pixels and saves the image
%         drawsuperpixelsonimage(A,iNumPixels,count,directory);

        aux = '';
        count = count + 1;
    end
