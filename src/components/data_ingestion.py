import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#from src.components.data_transformation import DataTransformationConfig,DataTransformation
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
            df=pd.read_csv("/Users/aryanpillai/PycharmProjects/Tarp_crop/notebook/APY.csv")
            logging.info("read csv dataset as df")
            df=df.dropna()
            sum_maxp = df["Production"].sum()
            df["percent_of_production"] = df["Production"].map(lambda x:(x/sum_maxp)*100)
            data1 = df.drop(["Crop_Year","District "],axis=1)
            data_dum = pd.get_dummies(data1)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            data_dum.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(data_dum,test_size=0.33,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data completed")

            return(
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__=='__main__':
    obj=DataIngestion()
    raw_data=obj.initiate_data_ingestion()
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(raw_data))

    # data_transformation = DataTransformation()
    # data_transformation.initiate_data_transformation(train_data,test_data)