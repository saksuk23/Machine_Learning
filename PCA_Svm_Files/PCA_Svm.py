import random
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # it is not an unused import... DO NOT DELETE!
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

NUM_OF_ASTERISKS = 50
NUM_OF_PCS = [3, 10, 20, 30]


# returns the mean correct predictions for the given confusion matrix
def get_mean_of_correct_predictions_matrix(matrix):
    return np.floor(np.mean([sum(np.diagonal(x)) for x in matrix]))


# printing method for all printing purposes
def print_stats_of_matrices(logistic_cm, knn_cm, svm_cm):
    print("__" * NUM_OF_ASTERISKS)
    print('The mean confusion matrix of the Logistic model is:\n {}'.format(get_mean_matrix(logistic_cm)))
    print("*" * NUM_OF_ASTERISKS)
    print('The mean confusion matrix of the KNN model is:\n {}'.format(get_mean_matrix(knn_cm)))
    print("*" * NUM_OF_ASTERISKS)
    print('The mean confusion matrix of the SVM model is:\n {}'.format(get_mean_matrix(svm_cm)))

    print("__" * NUM_OF_ASTERISKS)
    print('the mean correct predictions in the logistic model is: {}'.format(
        get_mean_of_correct_predictions_matrix(logistic_cm)))
    print("*" * NUM_OF_ASTERISKS)
    print('the mean correct predictions in KNN model is: {}'.format(get_mean_of_correct_predictions_matrix(knn_cm)))
    print("*" * NUM_OF_ASTERISKS)
    print('the mean correct predictions in SVM model is: {}'.format(get_mean_of_correct_predictions_matrix(svm_cm)))

    print("__" * NUM_OF_ASTERISKS)
    print('The Sensitivity of the Logistic model is: {0:.2f} '.format(get_sensitivity(logistic_cm) * 100))
    print("*" * NUM_OF_ASTERISKS)
    print('The Sensitivity of the KNN model is: {0:.2f}'.format(get_sensitivity(knn_cm) * 100))
    print("*" * NUM_OF_ASTERISKS)
    print('The Sensitivity of the SVM model is: {0:.2f}'.format(get_sensitivity(svm_cm) * 100))


# get the Sensitivity of a matrix
def get_sensitivity(matrix):
    true_positive_sum = []
    all_positive_predictions_sum = []
    for x in matrix:
        true_positive_sum.append(x[0][0])
        all_positive_predictions_sum.append(np.sum(x, axis=0)[0])
    return sum(true_positive_sum) / sum(all_positive_predictions_sum)


# get the mean value of a matrix
def get_mean_matrix(matrix):
    my_sum = 0
    for x in matrix:
        my_sum += x
    return (my_sum / 10).astype(int)


# create a vector for creating and calculating the linear combination
def get_vector_for_linear_comb(pca):
    lst = []
    for i in range(len(pca.components_[0])):
        lst.append(np.zeros(len(pca.components_[0])))
        lst[i][i] = 1
    return lst


# draw the graphs for the data
def draw_plots(x_train, y_train, pca):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for l, c, m in zip(range(0, 2), ('blue', 'red'), ('^', 'o')):
        ax.scatter(x_train[y_train == l, 0],
                   x_train[y_train == l, 1],
                   x_train[y_train == l, 2],
                   c=c,
                   label='class %s' % l,
                   alpha=0.5,
                   marker=m
                   )
    plt.show()

    x = get_vector_for_linear_comb(pca)
    for i in range(2):
        y = pca.components_[i]
        scalars = np.linalg.solve(x, y)
        plt.bar(range(len(scalars)), scalars, width=0.5, align="center")
        plt.show()


# main
def run_program():
    data_set = pd.read_csv("data1.csv", header=None)

    x = data_set.iloc[:, :-1].values
    y = data_set.iloc[:, -1].values

    x_train = None
    y_train = None
    pca = None

    logistic_cm = []
    knn_cm = []
    svm_cm = []
    rand_num_of_pc = NUM_OF_PCS[random.randint(0, 3)]
    print("The random num of pc generated is = {}".format(rand_num_of_pc))
    for index in range(10):
        # split train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        # scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Applying PCA
        pca = PCA(n_components=rand_num_of_pc)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

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

        # SVM
        svm_class = SVC(kernel='linear')
        svm_class.fit(x_train, y_train)
        y_predicted_svm = svm_class.predict(x_test)
        svm_cm.append(confusion_matrix(y_test, y_predicted_svm))

    print('The lambdas are = {} '.format(pca.explained_variance_))
    print('The explained variance = {} '.format(sum(pca.explained_variance_ratio_) * 100))

    print_stats_of_matrices(logistic_cm, knn_cm, svm_cm)
    draw_plots(x_train, y_train, pca)


# run the main and ignore warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    run_program()
