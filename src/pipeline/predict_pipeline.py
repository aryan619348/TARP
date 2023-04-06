import sys
import pandas as pd
from src.exception import CustomException
import os
from src.utils import load_object
import random
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            percent_of_production= float(1)
            Yield = random.uniform(0.5,1.5)
            #features.Area = features.Area.astype(float)
            features['percent_of_production']=percent_of_production
            features['Yield']= Yield
            features.rename(columns = {'Area':'Area '}, inplace = True)
            features = pd.get_dummies(features)
            test = pd.read_csv('artifacts/raw.csv')
            df_encoded = test.drop(columns=["Production"],axis=1)
            missing_columns = set(df_encoded.columns) - set(features.columns)
            for col in missing_columns:
                features[col] = 0

            features = features[df_encoded.columns]
            #features = features.drop(columns=features.columns[0])
            features.to_csv('artifacts/features_2.csv')
            model_path=os.path.join("artifacts","model.pkl")
            #preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            #preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            #data_scaled=preprocessor.transform(features)
            print(features.head())
            preds=model.predict(features)
            return preds

        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 State: str,
                 Crop: str,
                 Season: str,
                 Area: float):
        self.State=State
        self.Crop=Crop
        self.Season=Season
        self.Area=Area
        #self.Yield=Yield
        #self.percentage_of_production=State

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "State": [self.State],
                "Crop": [self.Crop],
                "Season": [self.Season],
                "Area": [self.Area]
            }

            df = pd.DataFrame(custom_data_input_dict)
            df.to_csv('artifacts/features.csv')
            return df
        except Exception as e:
            raise CustomException(e, sys)








