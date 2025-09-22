import numpy as np
import pandas as pd
import sys
from loggers import Logger
logger = Logger.get_logs('var_transform')

class Variable_Transform:
    def log_transform(x_train,x_test):
        try:
            logger.info('Variable Transformation has started..............')
            logger.info(f'first 10 samples in the x_train are : \n{x_train.iloc[:10, :]}')
            for i in x_train.columns:
                x_train[i+'_log'] = np.log(x_train[i]+1)
                x_test[i+'_log'] = np.log(x_test[i]+1)
            columns = []
            for j in x_train.columns:
                if '_log' not in j:
                    x_train = x_train.drop(j, axis=1)
                    x_test = x_test.drop(j, axis=1)
                    columns.append(j)
            logger.info(f'Columns we have removed from train and test are : {columns}')
            logger.info(f'Columns in the train data are : {x_train.columns}')
            logger.info(f'first 10 samples in the x_train are : \n{x_train.iloc[:10, :]}')
            logger.info(f'check null values:{x_train.isnull().sum()}')
            return x_train, x_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')