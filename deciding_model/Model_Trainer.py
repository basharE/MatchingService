import logging

import pandas as pd
import numpy as np

import xgboost as xgb
from bson import ObjectId

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn import tree, neighbors
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from configuration.ConfigurationService import get_database_uri_from_conf, get_database_name_from_conf, \
    get_database_collection_name_from_conf
from db.MongoConnect import connect_to_collection
from deciding_model.Db_to_df_converter import get_from_mongo_to_dataframe
from deciding_model.ResultEvaluator import ResultEvaluator


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ClassifierTrainer(metaclass=SingletonMeta):
    def __init__(self):
        self.best_classifier = None
        self.best_accuracy = 0
        self.classifiers = list()

    @staticmethod
    def perf_measure(y_actual, y_hat):
        _TP, _FP, _TN, _FN = 0, 0, 0, 0

        for actual, predicted in zip(y_actual, y_hat):
            if actual == predicted == 1:
                _TP += 1
            elif predicted == 1 and actual != predicted:
                _FP += 1
            elif actual == predicted == 0:
                _TN += 1
            elif predicted == 0 and actual != predicted:
                _FN += 1

        return f"TP: {_TP}, FP: {_FP}, TN: {_TN}, FN: {_FN}"

    def predict_all(self, input_df, _class):
        X = input_df.iloc[:, 2:]
        scaler = preprocessing.MinMaxScaler()
        X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        summed_probs = np.zeros((X_normalized.shape[0], 2))  # Adjust num_classes accordingly
        props_dic = dict()
        for clf in self.classifiers:
            probs = clf.predict_proba(X_normalized)
            clf_string = str(clf)
            result = clf_string.split("(", 1)[0]
            props_dic[result] = probs
            summed_probs += probs
            logging.info(f"Prediction Result of {clf.__class__.__name__} is {probs}")
        evaluator = ResultEvaluator()
        evaluator_res = evaluator.evaluate(props_dic, _class)
        if evaluator_res != None:
            id_of_object = input_df.loc[evaluator_res, 'id']
            document_id = ObjectId(id_of_object)

            collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
                                               get_database_collection_name_from_conf())
            document = collection.find_one({"_id": document_id},
                                           {'_id': 0, 'name': 1})

            if document:
                # Access the value of the specific field
                field_value = document.get("name")
                return field_value
            else:
                return None
        return evaluator_res

    def train_best_classifier(self, new):
        # Get dataframe from the database
        data_frame = get_from_mongo_to_dataframe(new)

        y = data_frame['class']
        X = data_frame.iloc[:, 2:]

        # Normalize data
        scaler = preprocessing.MinMaxScaler()
        X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y,
                                                            random_state=0,
                                                            test_size=0.3,
                                                            shuffle=True)

        # Define classifiers
        classifiers = [
            LogisticRegression(random_state=70),
            MultinomialNB(),
            tree.DecisionTreeClassifier(),

            RandomForestClassifier(),
            MLPClassifier(solver='lbfgs', hidden_layer_sizes=(20,), max_iter=1000),
            neighbors.KNeighborsClassifier(n_neighbors=3, weights="uniform"),
            AdaBoostClassifier(n_estimators=70, random_state=50),
            xgb.XGBClassifier(n_estimators=70, random_state=50),
            GaussianNB()
        ]

        num_folds = 10
        # Train and evaluate classifiers
        for clf in classifiers:
            # CROSS VALIDATION
            scores = cross_val_score(clf, X_normalized, y, cv=num_folds)
            # Calculate the mean accuracy across folds
            acc = scores.mean()
            logging.info(f"Cross-Validation Accuracy of {clf.__class__.__name__}: {scores}")
            logging.info(f"Mean Accuracy: {acc}")

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.classifiers.append(clf)
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
