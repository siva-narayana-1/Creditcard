import numpy as np
import pandas as pd
import sys
from sklearn.feature_selection import VarianceThreshold
from loggers import Logger
logger = Logger.get_logs('filter_methods')

class Filter_methods():
    def constant(x_train,x_test):
        try:
            con_v = VarianceThreshold(threshold=0)
            logger.info('Constant filter Constant method is applied....')
            con_v.fit(x_train)
            logger.info(f'Columns in the train data are : {len(x_train.columns)} -> After applyin the constant columns with variance not 0 are : {sum(con_v.get_support())} and variance with 0 are : {sum(~con_v.get_support())}')
            x_train = x_train.drop(x_train.columns[~con_v.get_support()], axis=1)
            x_test = x_test.drop(x_test.columns[~con_v.get_support()], axis=1)
            logger.info(f'Columns after droping are : {x_train.columns}')
            return x_train, x_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')

    def quasi_constant(x_train, x_test):
        try:
            qcon_v = VarianceThreshold(threshold=0.1)
            qcon_v.fit(x_train,x_test)
            logger.info('Constant filter Quasi Constant method is applied....')
            qcon_v.fit(x_train)
            logger.info(f'Columns in the train data are : {len(x_train.columns)} -> After applyin the constant columns with variance not 0 are : {sum(qcon_v.get_support())} and variance with 0 are : {sum(~qcon_v.get_support())}')
            x_train = x_train.drop(x_train.columns[~qcon_v.get_support()], axis=1)
            x_test = x_test.drop(x_test.columns[~qcon_v.get_support()], axis=1)
            logger.info(f'Columns after droping are : {x_train.columns}')
            return  x_train,x_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')
