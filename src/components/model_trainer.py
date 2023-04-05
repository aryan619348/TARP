import os
import sys

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models
#from src.components.data_ingestion import DataIngestionConfig,DataIngestion
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __int__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self,raw):
        try:
            test = pd.read_csv(raw)
           # test=test.reset_index()
            x= test.drop(columns=["Production"],axis=1)
            y= test[["Production"]]
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)
            model=Lasso()
            model.fit(x_train,y_train)
            preds = model.predict(x_test)
            # mean_squared_error(y_test,preds)
            # r2_score(y_test,preds)
            # model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,
            #                                  X_test=x_test,y_test=y_test,models=models)
            #
            # best_model_score=max(sorted(model_report.values()))
            #
            # best_model_name= max(sorted(model_report).keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            # best_model = models[best_model_name]
            # if best_model_score<0.6:
            #     raise CustomException("No best model")
            # logging.info("Done model")
            save_object(
                file_path='artifacts/model.pkl',
                obj=model
            )
            predicted = model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            logging.info("saved the lasso model")
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)




