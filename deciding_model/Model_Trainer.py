import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, tree, neighbors
from sklearn.metrics import f1_score
from deciding_model.Db_to_df_converter import get_from_mongo_to_dataframe


def perf_measure(y_actual, y_hat):
    TP, FP, TN, FN = 0, 0, 0, 0

    for actual, predicted in zip(y_actual, y_hat):
        if actual == predicted == 1:
            TP += 1
        elif predicted == 1 and actual != predicted:
            FP += 1
        elif actual == predicted == 0:
            TN += 1
        elif predicted == 0 and actual != predicted:
            FN += 1

    return f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}"


def train_model():
    # Get dataframe from the database
    data_frame = get_from_mongo_to_dataframe()

    # Define features and target variable
    features = ['image1_resnet', 'image1_clip', 'image2_resnet', 'image2_clip', 'image3_resnet', 'image3_clip',
                'image4_resnet',
                'image4_clip']
    X = data_frame.loc[:, features]
    y = data_frame.loc[:, ['class_of_image']]

    # Normalize data
    scaler = preprocessing.MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y,
                                                        random_state=0,
                                                        test_size=0.2,
                                                        shuffle=True)

    # Define classifiers
    classifiers = [
        svm.SVC(),
        tree.DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,)),
        neighbors.KNeighborsClassifier(n_neighbors=3, weights="uniform"),
        neighbors.KNeighborsClassifier(n_neighbors=3, weights="distance")
    ]

    # Train and evaluate classifiers
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy of {clf.__class__.__name__} is {acc}")

        y_random_result = data_frame.loc[X_test.index]
        y_random_result['class_pred'] = y_pred
        actual_classes = y_random_result['class_of_image'].tolist()
        predicted_classes = y_random_result['class_pred'].tolist()
        f1 = f1_score(actual_classes, predicted_classes, average='weighted')
        print(f"F1 Score of {clf.__class__.__name__} is {f1}")
