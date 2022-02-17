import pickle
from MachineLearning.config import data_folder
import numpy as np
import pandas as pd

missing_values = ['?', '--', ' ', 'NA', 'N/A', '-']
df = pd.read_csv(data_folder / "cardio_train.csv", delimiter=';', na_values=missing_values)


def clean(dataFrame = df):
    dataFrame.drop("id", axis=1, inplace=True)
    dataFrame.drop_duplicates(inplace=True)
    dataFrame["bmi"] = dataFrame["weight"] / (dataFrame["height"] / 100) ** 2   # create bmi column
    out_filter = ((dataFrame["ap_hi"] > 250) | (dataFrame["ap_lo"] > 200))
    dataFrame = dataFrame[~out_filter]

    out_filter2 = ((dataFrame["ap_hi"] < 0) | (dataFrame["ap_lo"] < 0))
    dataFrame = dataFrame[~out_filter2]

    cleaned_data = dataFrame.sample(frac=1)     # randomize data

    cleaned_train = cleaned_data.iloc[:68000, :]
    cleaned_test = cleaned_data.iloc[68000:, :]

    cleaned_test = cleaned_test[
        ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "bmi",
         "cardio"]]
    cleaned_test.to_csv(data_folder / "testingData.csv")
    print("written testing data")

    with open(data_folder / 'trainingData.dat', 'wb') as fh:
        pickle.dump(cleaned_train, fh)
        print("written training data")

    return cleaned_train


if __name__ == "__main__":
    clean(df)
