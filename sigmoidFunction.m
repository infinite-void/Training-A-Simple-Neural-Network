function [output] = sigmoidFunction(input);
  
%This function returns the value of the Activation function performed on the given.
%Here the activation function used is called a sigmoid function and it always 
%return a value between 0 and 1 for any given input.
  output = 1 ./ (1 + exp(-input));
endfunction
