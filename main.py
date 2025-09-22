import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from loggers import Logger
import warnings
warnings.filterwarnings('ignore')
logger = Logger.get_logs('main')
from transform import Variable_Transform
from missing_values import MissingData
from outliers_handle import Outliers
from filter_methods import Filter_methods
import matplotlib.pyplot as plt
from hyphotesis import Hyphotesis_test
from Encoding import Encode_data
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from algos import common
class CreditCard:
    def __init__(self):
        try:
            logger.info('--------------------Server started--------------------------')
            self.data = pd.read_csv('C:\\project\\Data\\creditcard.csv')
            logger.info(f'Data loaded sucessfully with shape of {self.data.shape}')
            logger.info(f'Check the null values in the dataset:{self.data.isnull().sum()}')
            self.data = self.data.dropna(subset=['NPA Status'], axis=0)
            self.data = self.data.drop('MonthlyIncome.1', axis=1)
            logger.info(f'shape of data set after droping Id: {self.data.shape}')
            logger.info(f'Check the null values in the dataset after dropna :{self.data.isnull().sum()}')
            self.X = self.data.iloc[:,:-1]
            self.y = self.data.iloc[:,-1]
            logger.info(f'Now we have divided the independent data and dependent columns seperately.')
            logger.info(f'Shape of independent columns is :{self.X.shape}')
            logger.info(f'Shape of dependent column is :{self.y.shape}')
            self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            logger.info(f'We have done the splitting and shpe of X_train is {self.X_train.shape} and X_test is {self.X_test.shape}')
            self.X_train_num= None
            self.X_tran_cat = None
            self.X_test_num = None
            self.X_test_cat = None
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')

    def feature_engineering(self):
        try:
            logger.info('-----------------Feature Engineering---------------------')
            logger.info(f'Send the train and test columns to handle the missing values.')
            self.X_train, self.X_test = MissingData.Random_sample(self.X_train, self.X_test)
            logger.info('Now we have to divide the numerical and categorical columns.')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info('we have divide the independent columns into numerical and categorical.')
            logger.info(f'Shape of Numerical columns are X_train{self.X_train_num.shape} and X_test{self.X_test_num.shape}')
            logger.info(f'{type(self.X_train_cat)}')
            logger.info(f'Shape of categorical columns are X_train{self.X_train_cat.shape} and X_test{self.X_test_cat.shape}')
            logger.info(f'Null values in Numerical columns are X_train: {self.X_train_num.isnull().sum().sum()} and X_test: {self.X_test_num.isnull().sum().sum()}')
            logger.info(f'Null values in categorical columns are X_train: {self.X_train_cat.isnull().sum().sum()} and X_test: {self.X_test_cat.isnull().sum().sum()}')
            logger.info(f'Null values in categorical columns are X_train: {self.X_train_cat.columns} and X_test: {self.X_test_cat.columns}')
            logger.info('Send the numerical columns to variable transformation')
            # Variable Tranformation
            self.X_train_num, self.X_test_num = Variable_Transform.log_transform(self.X_train_num, self.X_test_num)
            logger.info('We have done the variable transformation sucessfully.')
            logger.info(f'Check the null values in train : {self.X_train_num.isnull().sum()}')
            logger.info(f'Check the null values in test : {self.X_train_num.isnull().sum()}')
            #Outlier handling
            self.X_train_num, self.X_test_num = Outliers.trimming(self.X_train_num, self.X_test_num)
            # self.visual_outliers()
            logger.info('We have sucessfully done the feature engineering.')
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')

    def feature_selections(self):
        try:
            logger.info('-----------------Feature Selection---------------------')
            # Filter Methods
            self.X_train_num,self.X_test_num = Filter_methods.constant(self.X_train_num, self.X_test_num)
            self.X_train_num, self.X_test_num = Filter_methods.quasi_constant(self.X_train_num, self.X_test_num)
            logger.info(f'Sucessfully we have done filter method...')

            # Hyphotesis testing for finding co-varience and p-values
            self.X_train_num, self.X_test_num = Hyphotesis_test.hypo(self.X_train_num,self.X_test_num,self.y_train,self.y_test)
            logger.info('Sucessfully done the Hyphothesis...')
            # Encoding
            nominal_data_train = self.X_train_cat[['Gender', 'Region']]
            nominal_data_test = self.X_test_cat[['Gender', 'Region']]
            train_onencode, test_onencode = Encode_data.one_encoder(nominal_data_train, nominal_data_test)

            ordinal_data_train = self.X_train_cat.drop(['Gender', 'Region'], axis=1)
            ordinal_data_test = self.X_test_cat.drop(['Gender', 'Region'], axis=1)
            train_odencode, test_odencode = Encode_data.odinal_encoder(ordinal_data_train, ordinal_data_test)

            # Reset indexes
            train_onencode.reset_index(drop=True, inplace=True)
            test_onencode.reset_index(drop=True, inplace=True)
            train_odencode.reset_index(drop=True, inplace=True)
            test_odencode.reset_index(drop=True, inplace=True)
            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_test_num.reset_index(drop=True, inplace=True)

            # Combine categorical encodings
            self.training_data = pd.concat([self.X_train_num, train_onencode, train_odencode], axis=1)
            self.testing_data = pd.concat([self.X_test_num, test_onencode, test_odencode], axis=1)

            logger.info(f'Training encoded sample:\n{self.training_data.sample(5)}')
            logger.info(f'Testing encoded sample:\n{self.testing_data.sample(5)}')

            #Label encoding
            self.y_train, self.y_test = Encode_data.label_encode(self.y_train, self.y_test)
            logger.info(f'Sample dependent data : {self.y_train[:10]}')


        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')


    def balanced_data(self):
        try:
            logger.info('----------------Before Balancing------------------------')
            logger.info(f'Total row for Good category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 1)}')
            logger.info(f'Total row for Bad category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 0)}')
            logger.info(f'---------------After Balancing-------------------------')
            sm = SMOTE(random_state=42)
            self.training_data_res,self.y_train_res = sm.fit_resample(self.training_data,self.y_train)
            logger.info(f'Total row for Good category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 1)}')
            logger.info(f'Total row for Bad category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 0)}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')



    def feature_scaling(self):
        try:
            logger.info('---------Before scaling-------')
            logger.info(f'{self.training_data_res.head(4)}')
            logger.info(f'{self.training_data_res.columns}')
            sc = StandardScaler()
            sc.fit(self.training_data_res)
            self.training_data_res_t = sc.transform(self.training_data_res)
            self.testing_data_t = sc.transform(self.testing_data)
            logger.info('----------After scaling--------')
            logger.info(f'{self.training_data_res_t}')
            common(self.training_data_res_t, self.y_train_res, self.testing_data_t, self.y_test)
            logger.info(self.training_data_res.shape)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


    def visual_outliers(self):
        for i in self.X_train_num:
            sns.boxplot(x=self.X_train_num[i])
            plt.savefig(f'C:\\project\\outlier_img\\{i}.png')
            plt.show()
if __name__ == '__main__':
    obj = CreditCard()
    obj.feature_engineering()
    obj.feature_selections()
    obj.balanced_data()
    obj.feature_scaling()