This section is for the Logistic and Knn classifiers

The given program read a csv file taken from https://archive.ics.uci.edu/ml/datasets/Sports+articles+for+objectivity+analysis
and build 2 models for prediction: Logistic and KNN.
After building the modles and test them, the program outputs which model performed better in analyzing the data and made a better predictions of the test data

program flow:
Import the data and split it to X and y
Constructed 2 list that store the 10 confusion matrices
Loop 10 times to Split the data to train set and test set
Build and run a logistic model 
Build and run a KNN model
Print the mean confusion matrices
Print the mean correct predictions
Print the sensitivity of the model
