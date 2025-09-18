import numpy as np
import pandas as pd
import sys
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from loggers import Logger
logger = Logger.get_logs('hypothesis')

class Hyphotesis_test:
    def hypo(x_train, x_test, x_train_dep, x_test_dep):
        try:
            logger.info(f'Columns in the x_train are : {len(x_train.columns)}')
            logger.info(f'Columns in the x_test are : {len(x_test.columns)}')
            logger.info(f'Unique values in the target are:{x_train_dep.unique()}')
            x_train_dep = x_train_dep.map({'Good':1, 'Bad':0}).astype(int)
            c_values = []
            p_values = []
            for i in x_train.columns:
                c_value, p_value = pearsonr(x_train[i], x_train_dep)
                c_values.append(c_value)
                p_values.append(p_value)
            logger.info(f'Corelation values are : {c_values}')
            logger.info(f'P - values\n {p_values}')

            p_normal_index = []
            for i in range(len(p_values)):
                # value = float(p_values[i])
                normal_value = f'{format(float(p_values[i]),".80f")}'
                logger.info(f'{type(normal_value)}')
                if normal_value > '0.05':
                    p_normal_index.append(i)
            column = x_train.columns[p_normal_index]
            logger.info(f'Index value of the hyp > 0.05 are :{p_normal_index}')
            logger.info(f'Columns that has p value greather than 0.05 : {column}')
            x_train = x_train.drop(column, axis=1)
            x_test = x_test.drop(column, axis=1)
            logger.info(f'After removing the columns whose p value > 0.05 are : {x_train.columns}\n test :{x_test.columns}')
            return x_train, x_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')