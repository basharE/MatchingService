import logging

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, tree, neighbors
from sklearn.metrics import f1_score
from deciding_model.Db_to_df_converter import get_from_mongo_to_dataframe


class ClassifierTrainer:
    def __init__(self):
        self.best_classifier = None
        self.best_accuracy = 0

    @staticmethod
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

    def train_best_classifier(self, new):
        # Get dataframe from the database
        data_frame = get_from_mongo_to_dataframe(new)

        y = data_frame['class']
        X = data_frame.iloc[:, 1:]

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
            MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,), max_iter=1000),
            neighbors.KNeighborsClassifier(n_neighbors=3, weights="uniform"),
            neighbors.KNeighborsClassifier(n_neighbors=3, weights="distance")
        ]

        # Train and evaluate classifiers
        for clf in classifiers:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_classifier = clf

            y_random_result = data_frame.loc[X_test.index]
            y_random_result['class_pred'] = y_pred
            actual_classes = y_random_result['class'].tolist()
            predicted_classes = y_random_result['class_pred'].tolist()
            f1 = f1_score(actual_classes, predicted_classes, average='weighted')

            logging.info(f"Accuracy of {clf.__class__.__name__} is {acc}")
            logging.info(f"F1 Score of {clf.__class__.__name__} is {f1}")

        logging.info(f"Best classifier: {self.best_classifier.__class__.__name__} with accuracy: {self.best_accuracy}")
