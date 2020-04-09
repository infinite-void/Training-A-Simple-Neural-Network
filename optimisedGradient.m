function [weights cost] = optimisedGradient(rolledWeights, inputNodes, hiddenNodes, outputNodes, trainData, trainLabel, regParam, numberOfIterations)
%This function performs the gradient step of training model with help of 
%function fmincg() from Andrew Ng's Machine Learing Course on Coursera.org

%This set the options structure with maximum iterations to be passed to fmincg().
  options = optimset('MaxIter', numberOfIterations);

%Making a short-hand for the costFunction.
  costFunction1 = @(p) costFunction(p, inputNodes, hiddenNodes, outputNodes, trainData, trainLabel, regParam);

%Calling the fmincg() to perform the gradeint and return weights.
  [weights cost] = fmincg(costFunction1, rolledWeights, options);
endfunction
 
