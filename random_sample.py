import numpy as np
import pandas as pd
import sys
from loggers import Logs
logger = Logs.get_logger('random_sample')
class Random:
    def random_sample_imputation(x_train, x_test):
        try:
            f = []
            for i in x_train.columns:
                if x_train[i].isnull().sum() != 0:
                    if x_train[i].dtype == 'object':
                        x_train[i] = pd.to_numeric(x_train[i])
                        x_test[i] = pd.to_numeric(x_test[i])
                        f.append(i)
                    else:
                        f.append(i)

            for j in f:
                r_values = x_train[j].dropna().sample(x_train[j].isnull().sum(),random_state=42)
                r1_values = x_test[j].dropna().sample(x_test[j].isnull().sum(), random_state=42)
                r_values.index = x_train[x_train[j].isnull()].index
                r1_values.index = x_test[x_test[j].isnull()].index
                x_train[j+"_replaced"] = x_train[j].copy()
                x_test[j + "_replaced"] = x_test[j].copy()
                x_train.loc[x_train[j].isnull(),j+'_replaced'] = r_values
                x_test.loc[x_test[j].isnull(), j+'_replaced'] = r1_values
                x_train = x_train.drop([j], axis=1)
                x_test = x_test.drop([j], axis=1)
            logger.info(
                f"Sucessfully handled the missing values using random_sample imputation.\n{x_train.isnull().sum()} {x_test.isnull().sum()}")
            logger.info(f)
            return x_train,x_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

