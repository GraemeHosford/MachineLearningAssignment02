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

    sneakers_image_row_array = sneakers_feature.drop("label", inplace=False, axis=1).sample().to_numpy()
    boots_image_row_array = boots_feature.drop("label", inplace=False, axis=1).sample().to_numpy()

    print("\nTask 1 output:\n")
    print("\tThere are", len(sneakers_labels), "sneakers in this data")
    print("\tThere are", len(boots_labels), "ankle boots in this data")
    print("\tShowing an image for each feature class now...")

    # Must show image here for one of each
    plt.imshow(sneakers_image_row_array.reshape(28, 28))
    plt.show()

    plt.imshow(boots_image_row_array.reshape(28, 28))
    plt.show()

    # Must parameterise number of samples
    num_rows = 250

    sneakers_data = sneakers_feature.sample(num_rows)
    boots_data = boots_feature.sample(num_rows)

    return sneakers_data, boots_data


def task2(sneakers_data: pd.DataFrame, boots_data: pd.DataFrame):
    full_data = sneakers_data.append(boots_data)

    num_splits = 4
    for train_index, test_index in KFold(n_splits=num_splits, shuffle=True).split(full_data):
        train_fold = full_data.iloc[train_index]
        test_fold = full_data.iloc[test_index]

        pctrn = Perceptron()

        pctrn.fit(X=train_fold, y=train_fold)

        prediction = pctrn.predict(test_fold)

        accuracy = accuracy_score(test_fold, prediction)

        print("Accuracy for perceptron", accuracy * 100, "%")


def main():
    sneakers_data, boots_data = task1()
    task2(sneakers_data, boots_data)


main()
