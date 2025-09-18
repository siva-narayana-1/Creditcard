import numpy as np
import pandas as pd
import sys
from loggers import Logs
logger = Logs.get_logger('triming')
import warnings
warnings.filterwarnings('ignore')
class Trimmings:
    def trim(train_num, test_num):
        try:
            for i in train_num.columns:
                iqr = train_num[i].quantile(0.75) - train_num[i].quantile(0.25)
                upper_limit = train_num[i].quantile(0.75) + (1.5*iqr)
                lower_limit = train_num[i].quantile(0.25) - (1.5*iqr)
                train_num[i+'_trim'] = np.where(train_num[i] > upper_limit, upper_limit, np.where(train_num[i] < lower_limit, lower_limit, train_num[i]))
                test_num[i+'_trim'] = np.where(test_num[i] > upper_limit, upper_limit, np.where(test_num[i]<lower_limit, lower_limit, test_num[i]))
            logger.info(f"After triming column names:{train_num.columns}")
            logger.info(f"After triming column names:{test_num.columns}")

            columns = []
            for j in train_num.columns:
                if '_trim' not in j:
                    columns.append(j)
            train_num = train_num.drop(columns, axis=1)
            test_num = test_num.drop(columns, axis=1)
            return train_num, test_num
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error( f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")