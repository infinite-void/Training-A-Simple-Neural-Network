function [] = neuralNetworkDriver(trainData, trainLabel, numberofClasses, testData, testLabel)

%This function trains the model with given test data with help of 
%the function optimisedGradient and
%predicts the values for test data
%and gives the accuracy of the model.

%Get sizes of training and test data.
  trainDataSize = size(trainData, 1);
  testDataSize = size(testData, 1);

%Fix the number of nodes(neurons) in each layer of the network.
%Input layer has number of nodes equal to features in trianing set. MNIST
%contains 28*28 images and so 784 input nodes.
%Output nodes is the number of classes of data.(10 for digits 0-9)
%Hidden Nodes can be varied and is considered to be 800 here.
  inputNodes = size(trainData, 2);
  outputNodes = numberofClasses;
  hiddenNodes = 800;
 

%Regularisation parameter for the model can be varied.
  regularisationParameter = 1;

%Maximum iterations if the wieghts continue to converge further. 
  numberOfIterations = 200;
  
%Used to generate random weights in [0, epsilon];
  epsilon = 1;
  
%Generate random weights to train data using getRandomWeights.m. 
  fprintf('\nInitialising Random Weights for the Network...\n');
  rolledWeights = getRandomWeights( ((inputNodes + 1) * hiddenNodes) + ((hiddenNodes + 1) * outputNodes), epsilon);
  
%Training the neural network using optimisedGradient.m.
  fprintf('\nTraining the model using trainData...\n');
  [rolledWeights cost] = optimisedGradient(rolledWeights, inputNodes, hiddenNodes, outputNodes, trainData, trainLabel, regularisationParameter, numberOfIterations);
  
%predicting values for test data using predict.m
  fprintf('\nPredicting Values for testData...\n'); 
  predictedValues = predict(rolledWeights, testData, inputNodes, hiddenNodes, outputNodes);
  
%Calculate and display accuracy of the model on the test data set.  
  correctPredictions = sum(double(predictedValues == testLabel))
  fprintf('\nNumber Of Correct Predictions: %f\n', correctPredictions);
  fprintf('\nSize of Test Data: %f\n', testDataSize);
  fprintf('\nTraining Set Accuracy: %f\n', (correctPredictions / testDataSize) * 100);
  
endfunction
