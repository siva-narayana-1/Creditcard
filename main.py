import numpy as np
import pandas as pd
<<<<<<< HEAD
import sys
from sklearn.model_selection import train_test_split
from loggers import Logger
import warnings
warnings.filterwarnings('ignore')
logger = Logger.get_logs('main')
from missing_values import MissingData
from variable_transorm import Variable_Transform
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
            logger.info('-----------------Feature Selection---------------------')
            logger.info(f'Send the train and test columns to handle the missing values.')
            self.X_train, self.X_test = MissingData.Random_sample(self.X_train, self.X_test)
            logger.info('Now we have to divide the numerical and categorical columns.')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info('we have divide the independent columns into numerical and categorical.')
            logger.info(f'Shape of Numerical columns are X_train{self.X_train_num.shape} and X_test{self.X_test_num.shape}')
            logger.info(f'Shape of categorical columns are X_train{self.X_tran_cat.shape} and X_test{self.X_test_cat.shape}')
            logger.info(f'Null values in Numerical columns are X_train: {self.X_train_num.isnull().sum().sum()} and X_test: {self.X_test_num.isnull().sum().sum()}')
            logger.info(f'Null values in categorical columns are X_train: {self.X_tran_cat.isnull().sum().sum()} and X_test: {self.X_test_cat.isnull().sum().sum()}')
            logger.info(f'Null values in categorical columns are X_train: {self.X_tran_cat.columns} and X_test: {self.X_test_cat.columns}')
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

            # Combine categorical encodings
            self.training_data = pd.concat([train_onencode, train_odencode], axis=1)
            self.testing_data = pd.concat([test_onencode, test_odencode], axis=1)

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
            sc = StandardScaler()
            sc.fit(self.training_data_res)
            self.training_data_res_t = sc.transform(self.training_data_res)
            self.testing_data_t = sc.transform(self.testing_data)
            logger.info('----------After scaling--------')
            logger.info(f'{self.training_data_res_t}')
            common(self.training_data_res_t, self.y_train_res, self.testing_data_t, self.y_test)
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
=======
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


>>>>>>> b1bbf0c870a1d8ca10d3460eb94dab38a20b0b6a
