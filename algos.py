import numpy as np
import pandas as pd
import sys
import logging
from loggers import Logger
logger = Logger.get_logs('train_algo')
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def common(X_train, y_train, X_test, y_test):
    """Train, evaluate, and plot ROC curves for all classifiers in one plot"""
    try:
        classifiers = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(criterion='entropy'),
            "Random Forest": RandomForestClassifier(criterion='entropy', n_estimators=100)
        }

        plt.figure(figsize=(8, 6))

        for name, model in classifiers.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log metrics
            logger.info(f"------ {name} ------")
            logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            # Get probability scores for ROC
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except:
                y_prob = model.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            logger.info(f"AUC Score ({name}): {roc_auc}")
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.2f})")

        # Random guess lineq
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - All Models")
        plt.legend(loc="lower right")
        plt.show()

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue at {er_lin.tb_lineno} : {er_msg}')
