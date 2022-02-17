import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import scipy.stats as st
from sklearn.preprocessing import binarize
import seaborn as sn
from sklearn.metrics import roc_curve
from statsmodels.tools import add_constant
from MachineLearning.config import  data_folder


df = pd.read_csv("E:\project2\MachineLearning\cardio_train.csv", delimiter=';')


def BackFeatureElimination(data_frame, dep_var, col_list):
    while len(col_list) > 0:
        model = sm.Logit(dep_var, data_frame[col_list])
        result = model.fit(disp=0)
        largest_pvalue = round(result.pvalues, 3).nlargest(1)
        if largest_pvalue[0] < 0.05:
            return result, col_list
        else:
            col_list = col_list.drop(largest_pvalue.index)


def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        dataframe[feature].hist(bins=20, ax=ax, facecolor='midnightblue')
        ax.set_title(feature + " Distribution", color='DarkRed')

    fig.tight_layout()
    plt.savefig("histogram.png")

def reportDiagrams(heart_df=df):

    heart_df.drop(['id'], axis=1, inplace=True)
    heart_df = heart_df.replace({'gender': {1: 0, 2: 1}})
    #heart_df.drop(['education'], axis=1, inplace=True)
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("raw dataset heads without education")
    print(heart_df.head())

    draw_histograms(heart_df, heart_df.columns, 6, 3)

    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("count of no of people having heart disease and not having")
    print(heart_df.cardio.value_counts())

    #sn.countplot(x='cardio', data=heart_df)
    #plt.savefig("countplot.png")
    #sn.pairplot(data=heart_df)
    #plt.savefig("pairplot.png")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("dataset")
    print(heart_df.describe())

    heart_df_constant = add_constant(heart_df)
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("raw dataset head after adding constant")
    print(heart_df_constant.head())

    st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
    cols = heart_df_constant.columns[:-1]
    model = sm.Logit(heart_df.cardio, heart_df_constant[cols])
    result = model.fit()
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("model Summary")
    print(result.summary())

    result = BackFeatureElimination(heart_df_constant, heart_df.cardio, cols)[0]
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("summary after back feature elimination")
    print(result.summary())

    params = np.exp(result.params)
    conf = np.exp(result.conf_int())
    conf['OR'] = params
    pvalue = round(result.pvalues, 3)
    conf['pvalue'] = pvalue
    conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio', 'pvalue']
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("confidence")
    print((conf))

    new_features = heart_df[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol','gluc','smoke','alco','active', 'cardio']]
    x = new_features.iloc[:, :-1]
    y = new_features.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=5)

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)

    sklearn.metrics.accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
    plt.figure(figsize=(8, 5))
    sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
    plt.savefig("heatmap.png")

    TN = cm[0, 0]
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    sensitivity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)

    print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ', (TP + TN) / float(TP + TN + FP + FN), '\n',

          'The Missclassification = 1-Accuracy = ', 1 - ((TP + TN) / float(TP + TN + FP + FN)), '\n',

          'Sensitivity or True Positive Rate = TP/(TP+FN) = ', TP / float(TP + FN), '\n',

          'Specificity or True Negative Rate = TN/(TN+FP) = ', TN / float(TN + FP), '\n',

          'Positive Predictive value = TP/(TP+FP) = ', TP / float(TP + FP), '\n',

          'Negative predictive Value = TN/(TN+FN) = ', TN / float(TN + FN), '\n',

          'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ', sensitivity / (1 - specificity), '\n',

          'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ', (1 - sensitivity) / specificity)

    y_pred_prob = logreg.predict_proba(x_test)[:, :]
    y_pred_prob_df = pd.DataFrame(data=y_pred_prob,
                                  columns=['Prob of no heart disease (0)', 'Prob of Heart Disease (1)'])
    y_pred_prob_df.head()

    for i in range(1, 5):
        cm2 = 0
        y_pred_prob_yes = logreg.predict_proba(x_test)
        print(y_pred_prob_yes)
        y_pred2 = binarize(y_pred_prob_yes, threshold=i / 10)[:, 1]
        print(y_pred2)
        cm2 = confusion_matrix(y_test, y_pred2)
        print('With', i / 10, 'threshold the Confusion Matrix is ', '\n', cm2, '\n',
              'with', cm2[0, 0] + cm2[1, 1], 'correct predictions and', cm2[1, 0], 'Type II errors( False Negatives)',
              '\n\n',
              'Sensitivity: ', cm2[1, 1] / (float(cm2[1, 1] + cm2[1, 0])), 'Specificity: ',
              cm2[0, 0] / (float(cm2[0, 0] + cm2[0, 1])), '\n\n\n')

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:, 1])
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for Heart disease classifier')
    plt.xlabel('False positive rate (1-Specificity)')
    plt.ylabel('True positive rate (Sensitivity)')
    plt.grid(True)
    sklearn.metrics.roc_auc_score(y_test, y_pred_prob_yes[:, 1])
    plt.savefig("roc_curve.png")


def genReport():
    df = pd.read_csv(data_folder /  "cardio_train.csv", delimiter=';')


    df.drop("id", axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    out_filter = ((df["ap_hi"] > 250) | (df["ap_lo"] > 200))
    df = df[~out_filter]

    out_filter2 = ((df["ap_hi"] < 0) | (df["ap_lo"] < 0))
    dataFrame = df[~out_filter2]

    profile = ProfileReport(dataFrame, title="Pandas Profiling Report", explorative=True)
    profile.to_file(data_folder / "report.html")

    target_name = 'cardio'
    data_target = dataFrame[target_name]
    data = dataFrame.drop([target_name], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(data, data_target, test_size=.20, random_state=5)

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)

    sklearn.metrics.accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
    plt.figure(figsize=(8, 5))
    sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
    plt.savefig(data_folder / "heatmap.png")

    TN = cm[0, 0]
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    sensitivity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)

    print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ', (TP + TN) / float(TP + TN + FP + FN), '\n',

          'The Missclassification = 1-Accuracy = ', 1 - ((TP + TN) / float(TP + TN + FP + FN)), '\n',

          'Sensitivity or True Positive Rate = TP/(TP+FN) = ', TP / float(TP + FN), '\n',

          'Specificity or True Negative Rate = TN/(TN+FP) = ', TN / float(TN + FP), '\n',

          'Positive Predictive value = TP/(TP+FP) = ', TP / float(TP + FP), '\n',

          'Negative predictive Value = TN/(TN+FN) = ', TN / float(TN + FN), '\n',

          'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ', sensitivity / (1 - specificity), '\n',

          'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ', (1 - sensitivity) / specificity)

    y_pred_prob = logreg.predict_proba(x_test)[:, :]
    y_pred_prob_df = pd.DataFrame(data=y_pred_prob,
                                  columns=['Prob of no heart disease (0)', 'Prob of Heart Disease (1)'])
    y_pred_prob_df.head()

    for i in range(1, 5):
        cm2 = 0
        y_pred_prob_yes = logreg.predict_proba(x_test)
        print(y_pred_prob_yes)
        y_pred2 = binarize(y_pred_prob_yes, threshold=i / 10)[:, 1]
        print(y_pred2)
        cm2 = confusion_matrix(y_test, y_pred2)
        print('With', i / 10, 'threshold the Confusion Matrix is ', '\n', cm2, '\n',
              'with', cm2[0, 0] + cm2[1, 1], 'correct predictions and', cm2[1, 0], 'Type II errors( False Negatives)',
              '\n\n',
              'Sensitivity: ', cm2[1, 1] / (float(cm2[1, 1] + cm2[1, 0])), 'Specificity: ',
              cm2[0, 0] / (float(cm2[0, 0] + cm2[0, 1])), '\n\n\n')
        conf_matrix = pd.DataFrame(data=cm2, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
        plt.figure(figsize=(8, 5))
        sn.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
        plt.savefig(data_folder / "heatmapfinal.png")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:, 1])
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for Heart disease classifier')
    plt.xlabel('False positive rate (1-Specificity)')
    plt.ylabel('True positive rate (Sensitivity)')
    plt.grid(True)
    sklearn.metrics.roc_auc_score(y_test, y_pred_prob_yes[:, 1])
    plt.savefig(data_folder / "roc_curve.png")


if __name__ == "__main__":
    genReport()
