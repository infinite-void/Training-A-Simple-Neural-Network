%This file loads the data set into matrices and 
%and get the program to work by transferring control to
%the function neuralNetworkDriver.

%Loading the training data set(60000 examples)...
  fprintf('\nLoading Training Data...\n');
  mnist_train = csvread('mnist_train.csv');

%Loading the test data set(10000 examples)...  
  fprintf('\nLoading Test Data...\n');
  mnist_test = csvread('mnist_test.csv');

%Separating the feature of the data, the labels and the headers.
%Both mnist_train and mnist_test contain headers.
%First row is the header.
%First column(except (1,1)) are is the set of training labels.
  trainData = mnist_train(2:end, 2:end);
  trainLabel = mnist_train(2:end, 1:1);
  testData = mnist_test(2:end, 2:end);
  testLabel = mnist_test(2:end, 1:1);

%These two lines include the test data with the training data(70000 in total).
%This may get you a higher accuracy.
%Uncomment the lines to combine training and test data.
  
  trainData = [trainData; testData];
  trainLabel = [trainLabel; testLabel];
  
 
%Get the training and test data size.
  trainDataSize = size(trainLabel, 1);
  testDataSize = size(testLabel, 1);
  
  fprintf('\nSize of Training Data : %f\n', trainDataSize);
  fprintf('\nSize of Testing Data : %f\n', testDataSize);

%In the data set the digit zero has a label: 0.
%But as Octave is one-indexed the zeros are converted to tens
%in training and test data.
  for i=1:trainDataSize,
    if trainLabel(i)==0,
      trainLabel(i) = 10;
    endif;
  endfor;  

  for i=1:testDataSize,
    if testLabel(i)==0,
      testLabel(i) = 10;
    endif;
  endfor;

%As we classify digits 0-9 we have 10 classes  
  numberOfClasses = 10;

%Transferring control to the function neuralNetworkDriver.
  neuralNetworkDriver(trainData, trainLabel, numberOfClasses, testData, testLabel);
