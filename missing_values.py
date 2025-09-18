import numpy as np
import pandas as pd
import sys
from loggers import Logger
from cgitb import handler
logger = Logger.get_logs('missing_handle')

class MissingData:
    def Random_sample(X_train,X_test):
        try:
            colums = []
            for i in X_train.columns:
                if X_train[i].isnull().sum()!=0:
                    if X_train[i].dtype == 'object':
                        X_train[i] = pd.to_numeric(X_train[i])
                        X_test[i] = pd.to_numeric(X_test[i])
                        colums.append(i)
                    else:
                        colums.append(i)
            logger.info(f'Columns in the train and test are : {colums}')
            logger.info('start handle the missing values in the above columns')
            for j in colums:
                X_train_values = X_train[j].dropna().sample(X_train[j].isnull().sum(), random_state=42)
                X_test_values = X_test[j].dropna().sample(X_test[j].isnull().sum(), random_state=42)
                X_train_values.index = X_train[X_train[j].isnull()].index
                X_test_values.index = X_test[X_test[j].isnull()].index
                X_train[j+'_replaced'] = X_train[j].copy()
                X_test[j+'_replaced'] = X_test[j].copy()
                X_train.loc[X_train[j].isnull(), j+'_replaced'] = X_train_values
                X_test.loc[X_test[j].isnull(), j+'_replaced'] = X_test_values
                X_train = X_train.drop(j, axis=1)
                X_test = X_test.drop(j, axis=1)
            logger.info('We have sucessfully removed all null values in the train and test data.')
            logger.info(f'Check the null values:\n{X_train.isnull().sum()}')
            return X_train, X_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')