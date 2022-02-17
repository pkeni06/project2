import pandas as pd
import pickle
from MachineLearning.config import data_folder
modelName ="LogisticRegressionModel.dat"


def PredictResult(data):
    df = pd.DataFrame([data[1:13]],
                      columns=["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke",
                               "alco", "active", "bmi"])
    model = pickle.load(open(data_folder / modelName, 'rb'))
    return round(float(model.predict_proba(df)[:, 0] * 100), 2)


if __name__ == "__main__":
    data = [
        8645, 16568, 1, 170, 68.0, 140, 70, 1, 1, 0, 0, 0, 23.529411764705884, 0
    ]
    print(PredictResult(data))
