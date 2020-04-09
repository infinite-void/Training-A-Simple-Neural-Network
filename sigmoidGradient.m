function [output] = sigmoidGradient(input)
  
%This function return the derivative of the sigmoid function.
%sigmoid'(x) = sigmoid(x)(1-sigmoid(x)).

  output = sigmoidFunction(input) .* (1 - sigmoidFunction(input));
endfunction
