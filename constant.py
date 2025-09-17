import numpy as np
import pandas as pd
import sys
from sklearn.feature_selection import VarianceThreshold
reg = VarianceThreshold(threshold=0)
from loggers import Logs
logger = Logs.get_logger('constant')
class Constant_selection:
    def constant_s(train_num, test_num):
        try:
            reg.fit(train_num)
            logger.info(f"Sucessfully constant done.....")
            logger.info(f"Columns before varience:{train_num.shape[1]}--->Columns after applying the varienc not 0:{sum(reg.get_support())}--->Columns of variance O:{sum(~reg.get_support())}")
            logger.info(
                f"Columns before varience:{test_num.shape[1]}--->Columns after applying the varienc not 0:{sum(reg.get_support())}--->Columns of variance O:{sum(~reg.get_support())}")
            train_num = train_num.drop(train_num.columns[~reg.get_support()], axis=1)
            test_num = test_num.drop(test_num.columns[~reg.get_support()], axis=1)
            logger.info(f"Checking the columns after constant of variance threshold O are :\nTrain:{train_num.columns}\nTest:{test_num.columns}")
            return train_num,test_num
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
