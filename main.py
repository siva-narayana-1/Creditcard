import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from random_sample import Random
from log_transform import Transforms
from triming import Trimmings
from loggers import Logs
logger = Logs.get_logger("manin")
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
            for i in self.X_train_num:
                sns.boxplot(x=self.X_train_num[i])
                plt.savefig(f'C:\\credict_card\\outlier_img\\{i}.png')
                plt.show()
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

if __name__ == '__main__':
    try:
        obj = Creaditcard()
        obj.missing_values()
        obj.handle_outliers()
    except Exception as e:
        exc_type, exc_msg, exc_line = sys.exc_info()


