import numpy as np
import pandas as pd
from fontTools.merge.util import first
from sklearn.model_selection import train_test_split
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from random_sample import Random
from log_transform import Transforms
from triming import Trimmings
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from constant import Constant_selection
from hypothesis import Hypothesis_testing
from loggers import Logs
logger = Logs.get_logger("main")
import warnings
warnings.filterwarnings('ignore')
class Creaditcard:
    def __init__(self):
        try:
            self.df = pd.read_csv("C:\\credict_card\\creditcard.csv")
            logger.info(f"Loaded data sucessfully with shape of {self.df.shape}")
            logger.info(f"Null values in the data set are:{self.df.isnull().sum()}")
            self.df = self.df.drop([150000, 150001], axis=0)
            logger.info(f"After droping the rows in Good_Bad column ,Null values in the data set are:{self.df.isnull().sum()}")
            self.df = self.df.drop(["MonthlyIncome.1"], axis=1)
            logger.info(f"Loaded data sucessfully after droping MonthlyIncome.1 column with shape of {self.df.shape}")
            # Now divide the data as independent columns in X and dependent columns in y
            self.X = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]
            logger.info(f"After dividing the data as independent and dependent their shape is {self.X.shape} and {self.y.shape}")
            logger.info(f"sample X and y data are \n{self.X.sample(2)} \n {self.y.sample(2)}")
            # Now divide the data for training and testing
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=42)
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
    def missing_values(self):
        try:
            logger.info("missing values is activated...............")
            self.X_train, self.X_test = Random.random_sample_imputation(self.X_train, self.X_test)
            logger.info(f"After using random_sample imputation checking the null values \n {self.X_train.isnull().sum()}\n {self.X_test.isnull().sum()}")
            if self.X_train.isnull().sum().sum() == 0 and self.X_test.isnull().sum().sum() == 0:
                logger.info(f"We have checked their is no null values in the data set now.")
                # Now divide the data into two types numeric and category for both train and test
                self.X_train_num = self.X_train.select_dtypes(exclude='object')
                self.X_train_cat = self.X_test.select_dtypes(include='object')
                self.X_test_num = self.X_test.select_dtypes(exclude='object')
                self.X_test_cat = self.X_test.select_dtypes(include='object')
                logger.info(f"Sucessfully divided the train and test as numeric and category columns:\n train_num:{self.X_train_num.shape}\n train_cat:{self.X_train_cat.shape}\n test_num:{self.X_test_num.shape}\n test_cat:{self.X_test_cat.shape}")
            else:
                logger.info(f"We have checked their is null values in the data set.")
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
    def handle_outliers(self):
        try:
            logger.info("handle outlier is activated.......")
            self.X_train_num,self.X_test_num = Transforms.log_transformation(self.X_train_num, self.X_test_num)
            logger.info(f"shapes and data are :\n{self.X_train_num.shape}\nsample data are:\n{self.X_train_num.sample(5)}\n{self.X_test_num.shape}\nsample data are:\n{self.X_test_num.sample(5)}")
            self.X_train_num, self.X_test_num = Trimmings.trim(self.X_train_num, self.X_test_num)
            logger.info(f"We have done the trimming and now check the columns: \ntrain:\n{self.X_train_num.columns}\ntest:\n{self.X_test_num.columns}")
            # for i in self.X_train_num:
            #     sns.boxplot(x=self.X_train_num[i])
            #     plt.savefig(f'C:\\credict_card\\outlier_img\\{i}.png')
            #     plt.show()
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def feature_selection(self):
        try:
            self.X_train_num, self.X_test_num = Constant_selection.constant_s(self.X_train_num, self.X_test_num)
            logger.info(f'Before Hyphotesis train has:{len(self.X_train_num.columns)} columns')
            logger.info(f'Before Hyphotesis test has: {len(self.X_test_num.columns)} columns')
            self.X_train_num, self.X_test_num = Hypothesis_testing.hypo(self.X_train_num,self.X_test_num,self.y_train,self.y_test)
            logger.info(f'After Hyphotesis train has:{len(self.X_train_num.columns)} columns')
            logger.info(f'After Hyphotesis test has: {len(self.X_test_num.columns)} columns')
            #Nominal data  encoding
            logger.info("Encoding Started................")
            logger.info(f'{self.X_train_cat.columns}')
            nominal_train = self.X_train_cat[['Gender','Region']]
            nominal_test = self.X_test_cat[['Gender','Region']]
            oh = OneHotEncoder(categories='auto', drop='first', handle_unknown='error')
            oh.fit(nominal_train)
            logger.info(f'{oh.categories_}')
            logger.info(f'{oh.get_feature_names_out()}')
            nominal_train = oh.transform(nominal_train).toarray()
            nominal_test = oh.transform(nominal_test).toarray()
            logger.info(f'{nominal_train}\n{nominal_test}')
            f_train = pd.DataFrame(nominal_train, columns=oh.get_feature_names_out()+'_onc')
            f_test = pd.DataFrame(nominal_test, columns=oh.get_feature_names_out()+'_onc')
            logger.info(f'Sample data in the train is :\n{f_train}')
            logger.info(f'Sample data in the test is :\n{f_test}')
            #Concatenating the Encoded data and previous data
            # self.X_train_cat.reset_index(drop=True, inplace=True)
            # f_train.reset_index(drop=True,inplace=True)
            # self.X_test_cat.reset_index(drop=True, inplace=True)
            # f_test.reset_index(drop=True, inplace=True)
            # logger.info(f'Sample data in the train after concating :\n{self.X_train_cat}')
            # logger.info(f'Sample data in the test after concating :\n{self.X_test_cat}')
            # #OrdinalEncoder encoding
            # ordinal_train = self.X_train_cat.drop(['Gender','Region'], axis=1)
            # ordinal_test = self.X_test_cat.drop(['Gender', 'Region'], axis=1)
            # od = OneHotEncoder()
            # od.fit(ordinal_train)
            # logger.info(f'{od.categories_}')
            # logger.info(f'{od.get_feature_names_out()}')
            # ordinal_train = od.transform(ordinal_train).toarray()
            # ordinal_test = od.transform(ordinal_test).toarray()
            # logger.info(f'{ordinal_train}\n{ordinal_test}')
            # f1_train = pd.DataFrame(nominal_train, columns=od.get_feature_names_out()+'_enc')
            # f1_test = pd.DataFrame(nominal_test, columns=od.get_feature_names_out()+'_enc')
            # logger.info(f'Sample data in the train is :\n{f1_train}')
            # logger.info(f'Sample data in the test is :\n{f1_test}')
            # #Concatenating the Encoded data and previous data
            # self.X_train_cat.reset_index(drop=True, inplace=True)
            # f_train.reset_index(drop=True,inplace=True)
            # self.X_train_cat = pd.concat([self.X_train_cat,f1_train], axis=1)
            # self.X_test_cat.reset_index(drop=True, inplace=True)
            # f_test.reset_index(drop=True, inplace=True)
            # self.X_test_cat = pd.concat([self.X_test_cat, f1_test], axis=1)
            # logger.info(f'Sample data in the train after concating :\n{self.X_train_cat}')
            # logger.info(f'Sample data in the test after concating :\n{self.X_test_cat}')
            #
            # self.X_test_cat = pd.concat([self.X_test_cat, f_test], axis=1)
            # self.X_train_cat = pd.concat([self.X_train_cat, f_train], axis=1)
            # self.X_train_cat = self.X_train_cat.drop(['Gender','Region'], axis=1)
            # self.X_test_cat = self.X_test_cat.drop(['Gender', 'Region'], axis=1)
            # logger.info(f'Onhot encoding Sample data in the train after removing gender and region :\n{self.X_train_cat}')
            # logger.info(f'Onhot encoding Sample data in the test after removing gender and region :\n{self.X_test_cat}')

        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

if __name__ == '__main__':
    try:
        obj = Creaditcard()
        obj.missing_values()
        obj.handle_outliers()
        obj.feature_selection()
    except Exception as e:
        exc_type, exc_msg, exc_line = sys.exc_info()


