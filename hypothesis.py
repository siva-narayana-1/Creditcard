import numpy as np
import pandas as pd
import sys

from numpy.random import normal

from loggers import Logs
logger = Logs.get_logger('hypothesis')
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

class Hypothesis_testing:
    def hypo(train_num,test_num,train_num_dep,test_num_dep):
        try:
            logger.info(f'Train columns {len(train_num.columns)}')
            logger.info(f'Test columns {len(test_num.columns)}')
            logger.info(f'Unique values in the dependent column:{train_num_dep.unique()}')
            train_num_dep = train_num_dep.map({'Good':1,'Bad':0}).astype(int)
            logger.info(f'Unique values in the dependent column:{train_num_dep.unique()}')
            cp_values = []
            for i in train_num.columns: # passing the independennt columns and dependent columns to Pearsonr for getting co-relation and p-values.
                cp_values.append(pearsonr(train_num[i], train_num_dep))
            cp_values = np.array(cp_values)
            p_vals = cp_values[:, 1]
            val_list = []
            logger.info(f'ck:{type(p_vals)}, \n {p_vals}')
            normal_p = [format(float(val), ".80f") for val in p_vals]
            index_cols = []
            for i in range(len(normal_p)): # In this we are changing the numpy.float64 to float and get index whoes p value is grather than 0.05.
                normal_p[i] = float(normal_p[i])
                if normal_p[i] > 0.05:
                    index_cols.append(i)
            # print(train_num.columns[index_cols])
            l_columns = train_num.columns
            res = pd.Series(cp_values[:,1], index=l_columns) # setting p values and columns in series.
            columns_drop = train_num.columns[index_cols]
            logger.info(f'P_values of the train data is:\n{cp_values}')
            logger.info(f'P_values in normal (no scientific notation):\n{normal_p}')
            # logger.info(f'Index of max p-value is: {max_index}}')
            logger.info(f'Column that has high p value is : {columns_drop}')
            res = pd.Series(cp_values[:,1], index=train_num.columns)
            logger.info(f"columns wise P value is:{res}")
            res.plot.bar()
            plt.show()
            train_num = train_num.drop(columns_drop,axis=1)
            logger.info(f'Train columns {len(train_num.columns)}: {train_num.columns}')
            test_num = test_num.drop(columns_drop, axis=1)
            logger.info(f'Test columns {len(test_num.columns)}: {test_num.columns}')
            return train_num,test_num
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
