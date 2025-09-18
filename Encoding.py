import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from loggers import Logger
logger = Logger.get_logs('encoder')
import sys
class Encode_data:
    def one_encoder(x_train,x_test):
        try:
            logger.info("Encoding has started....")
            logger.info("------------------Onehot Encoder----------------------")
            logger.info(f'Columns in the x_ train are : {x_train.columns}')
            logger.info(f'Sample data are : {x_train}')
            oh = OneHotEncoder(categories='auto', drop='first', handle_unknown='ignore')
            oh.fit(x_train)
            logger.info(f'{oh.categories_}')
            logger.info(f'{oh.get_feature_names_out()}')
            output_train = oh.transform(x_train).toarray()
            logger.info(f'{output_train}')
            output_test = oh.transform(x_test).toarray()
            fd = pd.DataFrame(output_train, columns=oh.get_feature_names_out())
            fd_test = pd.DataFrame(output_test, columns=oh.get_feature_names_out())
            logger.info(f'Sample data after Onehotencoder is :\n{fd.sample(7)}')
            return fd, fd_test
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')

    def odinal_encoder(x_train1,x_test1):
        try:
            logger.info("------------------Ordinal Encoder-----------------------")
            logger.info(f'Columns in the x_ train are : {x_train1.columns}')
            logger.info(f'Sample data are : {x_train1}')
            od = OrdinalEncoder()
            od.fit(x_train1)
            logger.info(f'{od.categories_}')
            logger.info(f'{od.get_feature_names_out()}')
            output_train = od.transform(x_train1)
            output_test = od.transform(x_test1)
            columns = od.get_feature_names_out()
            fd1 = pd.DataFrame(output_train, columns=columns)
            fd_test1 = pd.DataFrame(output_test, columns=columns)
            logger.info(f'Sample data after Onehotencoder is :\n{fd1.sample(7)}')
            return fd1,fd_test1
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')

    def label_encode(train_dep,test_dep):
        try:
            lh = LabelEncoder()
            lh.fit(train_dep)
            res1 = lh.transform(train_dep)
            res2 = lh.transform(test_dep)
            logger.info(f'{res1}')
            logger.info(f'{lh.classes_}')
            return res1, res2
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')