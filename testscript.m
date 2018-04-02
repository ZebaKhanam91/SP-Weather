categories = {sprintf('Cloudy'), sprintf('Foggy'), sprintf('Rainy'), sprintf('Snowy'), sprintf('Sunny')};
    models = {sprintf('bvlc_googlenet'),sprintf('placesCNN'),sprintf('ResNet50'),sprintf('ResNet101'),sprintf('ResNet152'),sprintf('VGG_CNN_F'),sprintf('VGG_CNN_M'),sprintf('VGG_CNN_S'),sprintf('VGGNet16'),sprintf('VGGNet19') };
    NoOfCategories = size(categories);
    numsuperpixels = [25 50 75 100];
    C = 10;
    runid = 'a';
  DatasetPath = 'C:\Dataset\Cloudy\0001.jpg';
  I = imread(DatasetPath); 
  [L,N] = superpixels(I,25);
  BW = boundarymask(L);
  I = imoverlay(I,BW,'k');
  figure
  imshow(I,'InitialMagnification','fit')
   %imwrite(I,char(strcat(directory,pad(string(count),4,'left','0'),'.jpg')));