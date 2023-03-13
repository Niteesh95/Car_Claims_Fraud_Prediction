import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.train import ModelTrainConfig, train_model

@dataclass
class DataIngestionConfig:
    # train_data_path: str=os.path.join('artifacts', "train.csv")
    # test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "raw_data.csv")
    # transformed_data_path: str=os.path.join('artifacts', "transformed_data.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiated data ingestion")
        try:
            df=pd.read_csv('notebooks\data\carclaims.csv')
            logging.info("Dataset read complete")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            # train_set, test_set = train_test_split(df, test_size=0.2, random_state=15)

            # train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            # logging.info("Data ingestion completed")

            return (self.ingestion_config.raw_data_path)        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    raw_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_data, test_data = data_transformation.initiate_data_transformation(raw_data)

    model_trainer = train_model()
    print(model_trainer.initiateModelTrain(train_data,test_data))

