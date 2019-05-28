# Author: Amir Solnik
# Date: 5 Dec 2019

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import warnings


# returns the mean correct predictions for the given confusion matrix
def get_mean_of_correct_predictions_matrix(matrix):
    return np.floor(np.mean([sum(np.diagonal(x)) for x in matrix]))


# returns the sensitivity of the given confusion matrix
def get_sensitivity(matrix):
    true_positive_sum = []
    all_positive_predictions_sum = []
    for x in matrix:
        true_positive_sum.append(x[0][0])
        all_positive_predictions_sum.append(np.sum(x, axis=0)[0])
    return sum(true_positive_sum) / sum(all_positive_predictions_sum)


# return the mean value of the given confusion matrix
def get_mean_matrix(matrix):
    my_sum = 0
    for x in matrix:
        my_sum += x
    return (my_sum / 10).astype(int)


# reads and runs the program
def run_program():
	size = 10
	#read the csv file
    data_set = pd.read_csv("data1.csv", header=None)

	#split the dataset
    x = data_set.iloc[:, :-1].values
    y = data_set.iloc[:, -1].values

	#build the confusion matrices for logistic model and KNN model
    logistic_cm = []
    knn_cm = []
	
    for index in range(size):
        # split train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        # build and predict logistic regression
        log_regress = LogisticRegression(solver="lbfgs")
        log_regress.fit(x_train, y_train)
        y_predicted_log = log_regress.predict(x_test)
        logistic_cm.append(confusion_matrix(y_test, y_predicted_log))

        # KNN
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(x_train, y_train)
        y_predicted_knn = classifier.predict(x_test)
        knn_cm.append(confusion_matrix(y_test, y_predicted_knn))

    print('The mean confusion matrix of the Logistic model is:\n {}'.format(get_mean_matrix(logistic_cm)))
    print('The mean confusion matrix of the KNN model is:\n {}'.format(get_mean_matrix(knn_cm)))

    print('the mean correct predictions in the logistic model is: {}'.format(
        get_mean_of_correct_predictions_matrix(logistic_cm)))
    print('the mean correct predictions in KNN model is: {}'.format(get_mean_of_correct_predictions_matrix(knn_cm)))

    print('The Sensitivity of the logistic model is: {0:.2f} '.format(get_sensitivity(logistic_cm) * 100))
    print('The Sensitivity of the kn model is: {0:.2f}'.format(get_sensitivity(knn_cm) * 100))


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    run_program()
