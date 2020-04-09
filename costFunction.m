function [cost grad] = costFunction(rolledWeights, inputNodes, hiddenNodes, outputNodes, trainData, trainLabel, regParam)

%This function computes the regularised cost of the model for given set of weights.
%It also returns the gradient required adjust the weights of the model.

%The parameter rolledWeights contain all weights in a single vector.
%We reshape into weightsInputToHidden and weightsHiddenToOutput.
%weightsInputToHidden gets the data from input layer to hidden layer.
%weightsHiddenToOutput gets the data from hidden to output layer.
  weightsInputToHidden = reshape(rolledWeights(1:hiddenNodes * (inputNodes + 1)), hiddenNodes, (inputNodes + 1));
  weightsHiddenToOutput = reshape(rolledWeights((1 + (hiddenNodes * (inputNodes + 1))):end), outputNodes, (hiddenNodes + 1));

%Get size training data.
  trainDataSize = size(trainData, 1);

%Initialising the cost and gradients to be returned  
  cost = 0;
  gradWeightsInputToHidden = zeros(size(weightsInputToHidden));
  gradWeightsHiddenToOutput = zeros(size(weightsHiddenToOutput));

%Forward Propogation.
%Adding the bias feature to input layer.
  trainData = [ones(trainDataSize, 1) trainData];   
  
%Going from input layer to hidden layer.
  activationLayer = sigmoidFunction(trainData * weightsInputToHidden');   

%Adding the bias feature to hidden layer.  
  activationLayer = [ones(trainDataSize, 1) activationLayer]; 

%Going to output layer from hidden layer and getting the trained outputs. 
  trainOutput = sigmoidFunction(activationLayer * weightsHiddenToOutput');        
  
%As the train labels are present in a single vector. 
%Making the expectedOutput matix from train vector.  
  I = eye(outputNodes);          
  expectedOutput = zeros(trainDataSize, outputNodes);     
  for i = 1:trainDataSize,
      expectedOutput(i, :) = I(trainLabel(i), :);
  end

%Computing the unregularised cost.
  cost = - (1 / trainDataSize) * sum( sum( expectedOutput .* log(trainOutput) + (1 - expectedOutput) .* log(1 - trainOutput)));

%Removing the weights corresponding to bias features
%to compute regularisation parameters.
  tempWeightsInToHid = weightsInputToHidden;
  tempWeightsHidToOut = weightsHiddenToOutput;
  tempWeightsInToHid(:, 1) = 0;      
  tempWeightsHidToOut(:, 1) = 0;

%Computing the regularisation term and hence regularised cost.
  regularizationTerm = (regParam / (2 * trainDataSize)) * (sum(sum(tempWeightsInToHid .^ 2, 2)) + sum(sum(tempWeightsHidToOut .^ 2, 2)));
  cost = cost + regularizationTerm;

%BACK PROPOGATION!!!!!!!!!!!
%Computing error at output layer.
  backError3 = trainOutput - expectedOutput;

%Computing error at hidden layer and removing the values corresponding to bias feature.
  backError2 = (backError3 * weightsHiddenToOutput .* sigmoidGradient([ones(trainDataSize, 1) (trainData * weightsInputToHidden')]));
  backError2 = backError2(:, 2:end);

%Calculating the gradient at hidden layer.
  tempGrad1 = backError2' * trainData;   
%Calculating the gradient at input layer.
  tempGrad2 = backError3' * activationLayer;

%Computing the regularisedg gradients.
  gradWeightsInputToHidden = (1 / trainDataSize) .* tempGrad1 + (regParam/trainDataSize) * tempWeightsInToHid;
  gradWeightsHiddenToOutput = (1 / trainDataSize) .* tempGrad2 + (regParam/trainDataSize) * tempWeightsHidToOut;

%Rolling the gradients back into a single vector 
  grad = [gradWeightsInputToHidden(:) ; gradWeightsHiddenToOutput(:)];

endfunction
