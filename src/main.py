import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import warnings
warnings.filterwarnings('ignore')
from loggers import get_logger
logger = get_logger("main")
from multiprocessing import Pool
import threading
import time
import os
class IRIS:
    def __init__(self):
        try:
            logger.info("Server has strated >>>>>>>>>>.................")
            self.df = pd.read_csv('C:\\ML_all\\data\\Iris.csv')
            logger.info(f"Data loaded sucessfully with shape of {self.df.shape}")
            self.X = None
            self.y = None
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.training_data = pd.DataFrame()
            self.testing_data = pd.DataFrame()
            self.model = None
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
    def split_data(self):
        try:
            logger.info(f"Befor spliting the data checking the null values in dataset and remove the unnecessary column like Id in this,\n Null values:{self.df.isnull().sum().sum()}")
            self.df = self.df.drop('Id', axis=1)
            logger.info(f"Shape is after removing Id:{self.df.shape}")
            self.X = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
            logger.info(f"Shape of the independent columns are:{self.X.shape}\nShape of the dependent columns are:{self.y.shape}")
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=0.3, random_state=42)
            self.training_data = self.X_train
            self.training_data['y_actual'] = self.y_train
            self.testing_data = self.X_test
            self.testing_data['y_actual'] = self.y_test
            logger.info(f"Sucessfully divided the data for training and testing.\nShape of the training data is:{self.training_data.shape}\nShape of the testing data is:{self.testing_data.shape}")
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")
    def train_model(self, m, model_name):
        try:
            logger.info(f"----------------------Training the {model_name} Model-------------------------")
            logger.info(f"ThreadId:{threading.get_ident()}, ProcessorId:{os.getpid()}")
            time.sleep(2)
            self.model = m(criterion='entropy', max_depth=4, max_leaf_nodes=3)
            self.model.fit(self.X_train, self.y_train)
            logger.info("Sucessfully we have trained the model with entropy as criterion.")
            self.check_performance(self.X_train, self.y_train, type='training')
            self.check_performance(self.X_test, self.y_test, type='testing')
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def prediction(self, x):
        try:
            y_predict = self.model.predict(x)
            return y_predict
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def check_performance(self, x, y, type='training'):
        try:
            logger.info(f"----------------------Checking the performance of the {type}------------------------------")
            y_prediction = self.prediction(x)
            logger.info(f"Accuracy of the model is :{accuracy_score(y,y_prediction)}")
            logger.info(f'Confusion matrix of the model is :\n {confusion_matrix(y,y_prediction)}')
            logger.info(f'Classification Report of the model is :\n{classification_report(y,y_prediction)}')
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}")

    def thread(self, m):
        try:
            t = threading.Thread(target=self.train_model, args=(m,m.__name__))
            t.start()
            t.join()
        except Exception as e:
            exc_type, exc_msg, exc_line = sys.exc_info()
            logger.error(f"{exc_type} at {exc_line.tb_lineno} as {exc_msg}, {m}")


if __name__ == '__main__':
    obj = IRIS()
    obj.split_data()
    models = [DecisionTreeClassifier, RandomForestClassifier]
    with Pool(processes=2) as pool:
        p = pool.map(obj.thread, models)
