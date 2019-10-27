import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold


def task1():
    data = pd.read_csv("product_images.csv")

    sneakers_feature = data[data["label"] == 0]
    boots_feature = data[data["label"] == 1]

    sneakers_labels = sneakers_feature["label"]
    boots_labels = boots_feature["label"]

    sneakers_image_row = sneakers_feature.drop("label", inplace=False, axis=1).head(1).to_numpy()
    boots_image_row = boots_feature.drop("label", inplace=False, axis=1).head(1).to_numpy()

    print("\nTask 1 output:\n")
    print("\tThere are", len(sneakers_labels), "sneakers in this data")
    print("\tThere are", len(boots_labels), "ankle boots in this data")
    print("\tShowing images with matplotlib now...")

    # Must show image here for one of each
    plt.imshow(sneakers_image_row.reshape(28, 28))
    plt.show()

    plt.imshow(boots_image_row.reshape(28, 28))
    plt.show()

    # Must parameterise number of samples whatever that means

    return data


def task2(data: pd.DataFrame):
    num_splits = 4
    for train_index, test_index in KFold(n_splits=num_splits).split(data):
        train_fold = data.iloc[train_index]
        test_fold = data.iloc[test_index]


def main():
    data = task1()
    task2(data)


main()
