function predictedValues = predict(rolledWeights, testData, inputNodes, hiddenNodes, outputNodes)

%This function predicts the output values for given data and weights.
%Get size training data.
  testDataSize = size(testData, 1);

%The parameter rolledWeights contain all weights in a single vector.
%We reshape into weightsInputToHidden and weightsHiddenToOutput.
%weightsInputToHidden gets the data from input layer to hidden layer.
%weightsHiddenToOutput gets the data from hidden to output layer.
  weightInputToHidden = reshape(rolledWeights( 1:((inputNodes + 1) * hiddenNodes), :), hiddenNodes, inputNodes + 1);
  weightHiddenToOutput = reshape(rolledWeights( ((inputNodes + 1) * hiddenNodes) + 1:end, :), outputNodes, hiddenNodes + 1);

%Initialise the return value.
  predictedValues = zeros(testDataSize, 1); 

%Calculating hidden nodes from input nodes.
  hiddenActivation = sigmoidFunction([ones(testDataSize, 1) testData] * weightInputToHidden');
  
%Calculating the output from hidden layer.
  predictedData = sigmoidFunction([ones(testDataSize, 1) hiddenActivation] * weightHiddenToOutput');

%Find the class for which maximum probability was predicted. 
  [dummy predictedValues] = max(predictedData, [], 2);

endfunction
