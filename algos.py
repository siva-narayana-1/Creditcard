import numpy as np
import pandas as pd
import sys
import logging
from loggers import Logger
logger = Logger.get_logs('train_algo')
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


def knn_algo(X_train,y_train,X_test,y_test):
    try:
        knn_reg = KNeighborsClassifier(n_neighbors=5)
        knn_reg.fit(X_train,y_train)
        logger.info(f'Test Accuracy KNN : {accuracy_score(y_test,knn_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test,knn_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test,knn_reg.predict(X_test))}')
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def nb_algo(X_train,y_train,X_test,y_test):
    try:
        nb_reg = GaussianNB()
        nb_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy NB : {accuracy_score(y_test, nb_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, nb_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test, nb_reg.predict(X_test))}')
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def lr_algo(X_train,y_train,X_test,y_test):
    try:
        lr_reg = LogisticRegression()
        lr_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy LR : {accuracy_score(y_test, lr_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, lr_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test, lr_reg.predict(X_test))}')
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def dt_algo(X_train,y_train,X_test,y_test):
    try:
        dt_reg = DecisionTreeClassifier(criterion='entropy')
        dt_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy DT : {accuracy_score(y_test, dt_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, dt_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test, dt_reg.predict(X_test))}')
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def rf_algo(X_train,y_train,X_test,y_test):
    try:
        rf_reg = RandomForestClassifier(criterion='entropy',n_estimators=5)
        rf_reg.fit(X_train, y_train)
        logger.info(f'Test Accuracy RF : {accuracy_score(y_test, rf_reg.predict(X_test))}')
        logger.info(f'confusion matrix : {confusion_matrix(y_test, rf_reg.predict(X_test))}')
        logger.info(f'classification_report : {classification_report(y_test, rf_reg.predict(X_test))}')
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


def common(X_train,y_train,X_test,y_test):
    try:
        logger.info('Giving Data to Each Function')
        logger.info('----knn--------')
        knn_algo(X_train,y_train,X_test,y_test)
        logger.info('----NB--------')
        nb_algo(X_train, y_train, X_test, y_test)
        logger.info('----LR--------')
        lr_algo(X_train, y_train, X_test, y_test)
        logger.info('----dt--------')
        dt_algo(X_train, y_train, X_test, y_test)
        logger.info('----rf--------')
        rf_algo(X_train, y_train, X_test, y_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')