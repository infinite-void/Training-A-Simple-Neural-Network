# Training-A-Simple-Neural-Network.
Training a simple artificial neural network to identify had written digits with MNIST data set.

This repository is all about training a simple artificial neural network with the famous MNIST data set for handwritten digit recognition. 

The data set can be obtained from here : https://www.kaggle.com/c/digit-recognizer/data
In this data set, each image is represented in a 28x28 form.

The model has been based on the learnings from Machine Learning Course on Coursera by Andrew Ng. 

fmincg() :The file fmincg.m taken from the course material of that course and I totally credit them for the file. The file contains a function fmincg() which optimises the process of obtaining weights. 

Trying with other data sets :
You can simply try this model with other datasets just by replacing filenames and altering the trainData and testData matrices in the file loadAndDeploy.m. 

Altering the hidden layer : 
The number of nodes in the hidden layer can be altered from the file neuralNetworkDriver.m.

Altering the regularisation parameter :
The regularistion parameter can be altered in neuralNetwokDriver.m and should be set to zero in case of nil regularisation(this may lead to overfitting the data).

Accuracy of the model : 
The model achieves an accuracy of about 95% with the given 60000 trianing examples in train_mnist.csv and 10000 test examples in test_mnist.csv.
The model achieves an accuracy of about 99% by combining the training and test data together(70000 examples in total) for training and testing on the test set. This idea was given here : https://www.kaggle.com/c/digit-recognizer/discussion/61480.

