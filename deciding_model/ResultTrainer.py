import logging
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import tree, neighbors


class ResultTrainer:
    _instance = None  # Private class variable to store the instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResultTrainer, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.classifiers = list()

    def train_best_classifier(self, data_frame):
        y = data_frame['class']
        X = data_frame.drop(columns=['class'])

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
            tree.DecisionTreeClassifier(),
            RandomForestClassifier(),
            MLPClassifier(solver='lbfgs', hidden_layer_sizes=(20,), max_iter=1000),
            neighbors.KNeighborsClassifier(n_neighbors=3, weights="uniform"),
            AdaBoostClassifier(n_estimators=70, random_state=50),
            # XGBClassifier(n_estimators=70, random_state=50),
            LogisticRegression(random_state=70),
            MultinomialNB(),
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

            y_random_result = data_frame.loc[X_test.index]
            y_random_result['class_pred'] = y_pred
            actual_classes = y_random_result['class'].tolist()
            predicted_classes = y_random_result['class_pred'].tolist()
            f1 = f1_score(actual_classes, predicted_classes, average='weighted')

            logging.info(f"Accuracy of {clf.__class__.__name__} is {acc}")
            logging.info(f"F1 Score of {clf.__class__.__name__} is {f1}")

        logging.info("Best")
