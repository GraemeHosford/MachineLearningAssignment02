import timeit

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model.perceptron import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def task1():
    data = pd.read_csv("product_images.csv")

    sneakers_feature = data[data["label"] == 0]
    boots_feature = data[data["label"] == 1]

    sneakers_labels = sneakers_feature["label"]
    boots_labels = boots_feature["label"]

    sneakers_feature.drop("label", inplace=True, axis=1)
    boots_feature.drop("label", inplace=True, axis=1)

    # Must parameterise number of samples
    num_rows = 250

    sneakers_data = sneakers_feature.sample(num_rows)
    boots_data = boots_feature.sample(num_rows)

    sneakers_label_data = sneakers_labels.sample(num_rows)
    boots_label_data = boots_labels.sample(num_rows)

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

    return sneakers_data, boots_data, sneakers_label_data, boots_label_data


def task2(sneakers_data: pd.DataFrame, boots_data: pd.DataFrame,
          sneakers_labels: pd.DataFrame, boots_labels: pd.DataFrame):
    full_data = sneakers_data.append(boots_data)
    full_labels = sneakers_labels.append(boots_labels)

    num_splits = 4
    for train_index, test_index in KFold(n_splits=num_splits, shuffle=True).split(full_data):
        train_data = full_data.iloc[train_index]
        test_data = full_data.iloc[test_index]

        train_labels = full_labels.iloc[train_index]
        test_labels = full_labels.iloc[test_index]

        pctrn = Perceptron()

        pctrn.fit(X=train_data, y=train_labels)

        prediction = pctrn.predict(test_data)

        accuracy = accuracy_score(test_labels, prediction)

        print("\tTask 2 output")
        print("\tAccuracy for perceptron", accuracy * 100, "%")
        print("\n")


def task3():
    print("SVM Next")


def main():
    sneakers_data, boots_data, sneakers_labels, boots_labels = task1()

    task_2_start_time = timeit.default_timer()
    task2(sneakers_data, boots_data, sneakers_labels, boots_labels)
    task_2_stop_time = timeit.default_timer()

    task_2_runtime = task_2_stop_time - task_2_start_time

    print("\tTask 2 took", task_2_runtime, "seconds to run")

    task_3_start_time = timeit.default_timer()

    task_3_stop_time = timeit.default_timer()

    task_3_runtime = task_3_stop_time - task_3_start_time

    print("\tTask 3 took", task_3_runtime, "seconds to run")


main()
