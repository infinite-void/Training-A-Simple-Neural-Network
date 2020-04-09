function [output] = getRandomWeights(input, epsilon)

%This function simply generates random weights for the model 
%to begin with.

  output = ((2 * epsilon) * rand(input, 1)) - epsilon;
endfunction
