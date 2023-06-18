import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, tree, neighbors
from sklearn.metrics import f1_score
from deciding_model.Db_to_df_converter import get_from_mongo_to_dataframe


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    return "TP: " + str(TP) + ", FP: " + str(FP) + ", TN: " + str(TN) + ", FN: " + str(FN)


def train_model():
    # get dataframe from db
    data_frame = get_from_mongo_to_dataframe()
    features = ['image1_resnet', 'image1_clip', 'image2_resnet', 'image2_clip', 'image3_resnet', 'image3_clip',
                'image4_resnet',
                'image4_clip']
    X = data_frame.loc[:, features]
    y = data_frame.loc[:, ['class_of_image']]

    # normalizing data
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    X_normalized = pd.DataFrame(x_scaled)

    # splitting data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y,
                                                        random_state=0,
                                                        test_size=0.2,
                                                        shuffle=True)
    # build classifiers
    classifiers = list()
    n_neighbors = 3

    model1 = xgboost.XGBClassifier()
    classifiers.append(model1)
    model2 = svm.SVC()
    classifiers.append(model2)
    model3 = tree.DecisionTreeClassifier()
    classifiers.append(model3)
    model4 = RandomForestClassifier()
    classifiers.append(model4)
    model5 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,))
    classifiers.append(model5)
    model6 = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
    classifiers.append(model6)
    model7 = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
    classifiers.append(model7)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy of %s is %s" % (clf, acc))

        print("F1 Score of %s is" % clf)
        y_random_result = data_frame.loc[X_test.index]
        y_random_result['class_pred'] = y_pred
        print(perf_measure(y_random_result['class'].tolist(), y_random_result['class_pred'].tolist()))
        print(f1_score(y_random_result['class'].tolist(), y_random_result['class_pred'].tolist(), average='weighted'))
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix of %s is" % clf)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=clf.classes_)
        disp.plot()
        plt.show()
