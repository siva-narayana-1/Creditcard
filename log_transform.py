'''
In this file we have creating a function for variable transformation for numeric data in the data set.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loggers import Logs
logger = Logs.get_logger("log_transform")
from transform_visuals import Visuals
import sys
class Transforms:
    def log_transformation(x_train, x_test):
        try:
            for i in x_train.columns:
                x_train[i+'_log'] = np.log(x_train[i]+1)
                x_test[i+'_log'] = np.log(x_test[i]+1)
            logger.info(f"Transformed sucessfully")
            columns = []
            for j in x_train.columns:
                if '_log' not in j:
                    columns.append(j)
            x_train = x_train.drop(columns, axis=1)
            x_test = x_test.drop(columns, axis=1)
            logger.info(f"Sample data after removing the non _log columns:\nX_train:\n{x_train}\nX_test:\n{x_test}")
            return x_train,x_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
