import pandas as pd
import pickle
import MachineLearning.Data as Data
from sklearn.linear_model import LogisticRegression
from MachineLearning.config import data_folder

# missing_values = ['?', '--', ' ', 'NA', 'N/A', '-']
# data = pd.read_csv("trainingData.csv", delimiter=';', na_values=missing_values)
try:
    data = pd.read_pickle(data_folder / "trainingData.dat")
except Exception as e:
    if type(e) == FileNotFoundError:
        print("Cleaning Data")
        data = Data.clean()
    else:
        print(e)

def trainModel(cleaned_data=data):
    #    X = cleaned_data.drop(['cardio', 'bmi', 'weight', 'gluc', 'gender', 'smoke', 'alco', 'active'], axis=1)
    X = cleaned_data.drop(['cardio'], axis=1)
    Y = cleaned_data['cardio']

    logisticModel = LogisticRegression(max_iter=1000)
    logisticModel.fit(X, Y)

    with open(data_folder / 'LogisticRegressionModel.dat', 'wb') as fh:
        pickle.dump(logisticModel, fh)


def loadModel(fileName=data_folder / "LogisticRegressionModel.dat"):
    model = pickle.load(open(fileName, 'rb'))
    return model


if __name__ == "__main__":
    trainModel(data)
