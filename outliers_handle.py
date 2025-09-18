import numpy as np
import pandas as pd
import sys
from loggers import Logger
logger = Logger.get_logs('outliers')

class Outliers:
    def trimming(x_train,x_test):
        try:
            logger.info('Outlier has started ............')
            logger.info(f'Columns in the train are : {x_train.columns}')
            for i in x_train.columns:
                iqr = x_train[i].quantile(0.75) - x_train[i].quantile(0.25)
                upper_limit = x_train[i].quantile(0.75) + (1.5* iqr)
                lower_limit = x_train[i].quantile(0.25) - (1.5 * iqr)
                x_train[i+'_trim'] = np.where(x_train[i]>upper_limit, upper_limit, np.where(
                    x_train[i] < lower_limit, lower_limit, x_train[i]
                ))
                x_test[i+'_trim'] = np.where(x_test[i]>upper_limit, upper_limit, np.where(
                    x_test[i] < lower_limit, lower_limit, x_test[i]))

            for j in x_train.columns:
                if '_trim' not in j:
                    x_train = x_train.drop(j, axis=1)
                    x_test = x_test.drop(j, axis=1)
            logger.info(f'Columns in the train are : {x_train.columns}')
            return x_train,x_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')