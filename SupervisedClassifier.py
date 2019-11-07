import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model.perceptron import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold


def task1(num_rows: int):
    data = pd.read_csv("product_images.csv")

    sampled_data = data.sample(num_rows)

    sneakers_feature = sampled_data[sampled_data["label"] == 0]
    boots_feature = sampled_data[sampled_data["label"] == 1]

    sneakers_labels = sneakers_feature["label"]
    boots_labels = boots_feature["label"]

    sneakers_feature.drop("label", inplace=True, axis=1)
    boots_feature.drop("label", inplace=True, axis=1)

    sneakers_image_row_array = sneakers_feature.sample().to_numpy()
    boots_image_row_array = boots_feature.sample().to_numpy()

    print("\nTask 1 output:\n")
    print("\tThere are", len(sneakers_labels), "sneakers in this data")
    print("\tThere are", len(boots_labels), "ankle boots in this data")
    print("\tShowing an image for each feature class now...")
    print("\n\n")

    plt.gray()

    plt.imshow(sneakers_image_row_array.reshape(28, 28))
    plt.show()

    plt.imshow(boots_image_row_array.reshape(28, 28))
    plt.show()

    return sneakers_feature, boots_feature, sneakers_labels, boots_labels


def task2(sneakers_data: pd.DataFrame, boots_data: pd.DataFrame,
          sneakers_labels: pd.DataFrame, boots_labels: pd.DataFrame):
    full_data = sneakers_data.append(boots_data)
    full_labels = sneakers_labels.append(boots_labels)

    train_times = []
    predict_times = []
    accuracies = []

    print("\tTask 2 output")
    num_splits = 4
    for train_index, test_index in KFold(n_splits=num_splits, shuffle=True).split(full_data):
        train_data = full_data.iloc[train_index]
        test_data = full_data.iloc[test_index]

        train_labels = full_labels.iloc[train_index]
        test_labels = full_labels.iloc[test_index]

        pctrn = Perceptron()

        train_start_time = timeit.default_timer()
        pctrn.fit(X=train_data, y=train_labels)
        train_end_time = timeit.default_timer()

        train_time = train_end_time - train_start_time
        train_times.append(train_time)

        print("\tPerceptron took", train_time, "seconds to train on data")

        predict_start_time = timeit.default_timer()
        prediction = pctrn.predict(test_data)
        predict_end_time = timeit.default_timer()

        predict_time = predict_end_time - predict_start_time
        predict_times.append(predict_time)

        print("\tPerceptron took", predict_time, "seconds to make a prediction")

        accuracy = accuracy_score(test_labels, prediction) * 100
        accuracies.append(accuracy)

        print("\tAccuracy for perceptron", accuracy, "%")

        confusion = confusion_matrix(test_labels, prediction)

        percent_true_pos = (confusion[0, 0] / len(test_labels)) * 100
        percent_false_pos = (confusion[0, 1] / len(test_labels)) * 100
        percent_true_neg = (confusion[1, 1] / len(test_labels)) * 100
        percent_false_neg = (confusion[1, 0] / len(test_labels)) * 100

        print("\tPerceptron confusion matrix true positive:", percent_true_pos, "%")
        print("\tPerceptron confusion matrix false positive:", percent_false_pos, "%")
        print("\tPerceptron confusion matrix true negative:", percent_true_neg, "%")
        print("\tPerceptron confusion matrix false negative:", percent_false_neg, "%\n")

    print("\tThe minimum train time was", np.min(train_times), "seconds")
    print("\tThe maximum train time was", np.max(train_times), "seconds")
    print("\tThe average train time was", np.average(train_times), "seconds\n")

    print("\tThe minimum prediction time was", np.min(predict_times), "seconds")
    print("\tThe maximum prediction time was", np.max(predict_times), "seconds")
    print("\tThe average prediction time was", np.average(predict_times), "seconds\n")

    print("\tAverage accuracy was", np.average(accuracies), "%\n")


def task3(sneakers_data: pd.DataFrame, boots_data: pd.DataFrame,
          sneakers_labels: pd.DataFrame, boots_labels: pd.DataFrame, kernel_type: str):
    full_data = sneakers_data.append(boots_data)
    full_labels = sneakers_labels.append(boots_labels)

    train_times = []
    predict_times = []
    accuracies = []

    print("\tTask 3 output")
    num_splits = 4
    for train_index, test_index in KFold(n_splits=num_splits, shuffle=True).split(full_data):
        train_data = full_data.iloc[train_index]
        test_data = full_data.iloc[test_index]

        train_labels = full_labels.iloc[train_index]
        test_labels = full_labels.iloc[test_index]

        clf = svm.SVC(kernel=kernel_type, gamma=1e-3)

        train_start_time = timeit.default_timer()
        clf.fit(train_data, train_labels)
        train_end_time = timeit.default_timer()

        train_time = train_end_time - train_start_time

        train_times.append(train_time)

        print("\tTraining for", kernel_type, "SVM took", train_time, "seconds")

        predict_start_time = timeit.default_timer()
        prediction = clf.predict(test_data)
        predict_end_time = timeit.default_timer()

        predict_time = predict_end_time - predict_start_time
        predict_times.append(predict_time)

        print("\tPredicting for the", kernel_type, "SVM took", predict_time, "seconds")

        acc_score = accuracy_score(test_labels, prediction) * 100
        accuracies.append(acc_score)

        print("\tAccuracy for", kernel_type, "SVM", acc_score, "%")

        confusion = confusion_matrix(test_labels, prediction)

        percent_true_pos = (confusion[0, 0] / len(test_labels)) * 100
        percent_false_pos = (confusion[0, 1] / len(test_labels)) * 100
        percent_true_neg = (confusion[1, 1] / len(test_labels)) * 100
        percent_false_neg = (confusion[1, 0] / len(test_labels)) * 100

        print("\tPerceptron confusion matrix true positive:", percent_true_pos, "%")
        print("\tPerceptron confusion matrix false positive:", percent_false_pos, "%")
        print("\tPerceptron confusion matrix true negative:", percent_true_neg, "%")
        print("\tPerceptron confusion matrix false negative:", percent_false_neg, "%\n")

    print("\tThe minimum train time was", np.min(train_times), "seconds")
    print("\tThe maximum train time was", np.max(train_times), "seconds")
    print("\tThe average train time was", np.average(train_times), "seconds\n")

    print("\tThe minimum prediction time was", np.min(predict_times), "seconds")
    print("\tThe maximum prediction time was", np.max(predict_times), "seconds")
    print("\tThe average prediction time was", np.average(predict_times), "seconds\n")

    print("\tAverage accuracy was", np.average(accuracies), "%\n")


def main():
    for num_rows in [500, 1000, 3000]:
        print("Output for 500 rows")
        sneakers_data, boots_data, sneakers_labels, boots_labels = task1(num_rows)

        task2(sneakers_data, boots_data, sneakers_labels, boots_labels)

        task3(sneakers_data, boots_data, sneakers_labels, boots_labels, "linear")
        task3(sneakers_data, boots_data, sneakers_labels, boots_labels, "rbf")


main()
