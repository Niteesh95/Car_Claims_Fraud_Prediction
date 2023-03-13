import sys
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object
# from src.components.data_ingestion import DataIngestionConfig,DataIngestion

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    transformed_data_path: str=os.path.join('artifacts', "transformed_data.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, raw_data_path):
        try:
            # ingestion_config = DataIngestionConfig()
            train_df = pd.read_csv(raw_data_path)
            # test_df = pd.read_csv(test_path)
            logging.info("Data read completed - data transformation")

            # train_df.to_csv(ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Data Cleaning and Feature Engineering")
            target_column = 'FraudFound'

            train_df['Age_Group'] = pd.cut(train_df['Age'],
                                     bins=[0,9,19,29,39,49,59,69,79,89],
                                     labels=["0-9", "10-19","20-29","30-39","40-49","50-59", "60-69","70-79","80-89"])
            train_df.drop(['WeekOfMonth', 'DayOfWeek', "Make", 'DayOfWeekClaimed', 'MonthClaimed', 'WeekOfMonthClaimed', 'PolicyNumber',
                     'RepNumber', 'PastNumberOfClaims', 'DriverRating', 'Year', 'Month', 'VehicleCategory', 'BasePolicy'], inplace=True, axis=1)
            
            num_cols = train_df[['Age', 'Deductible']]
            cat_cols = train_df.loc[:, ~train_df.columns.isin(['Age', 'Deductible'])]
            # replacing 1 and 0 for above features
            cat_cols['AccidentArea'] = cat_cols['AccidentArea'].map(dict(Urban=1, Rural=0))
            cat_cols['Sex'] = cat_cols['Sex'].map(dict(Male=1, Female=0))
            cat_cols['PoliceReportFiled'] = cat_cols['PoliceReportFiled'].map(dict(Yes=1, No=0))
            cat_cols['WitnessPresent'] = cat_cols['WitnessPresent'].map(dict(Yes=1, No=0))

            cat_cols['Fault'].replace(['Policy Holder', 'Third Party'],[1, 0], inplace=True)
            cat_cols['AgentType'].replace(['External', 'Internal'],[1, 0], inplace=True)
            cat_cols['FraudFound'].replace(['No', 'Yes'],[0, 1], inplace=True)
            cat_dummies = pd.get_dummies(cat_cols[['MaritalStatus', 'PolicyType', 'VehiclePrice', 'Days:Policy-Accident', 'Days:Policy-Claim',
                        'AgeOfVehicle', 'NumberOfSuppliments', 'AddressChange-Claim', 'NumberOfCars', 'Age_Group']])
            a = cat_cols[['AccidentArea', 'Sex', 'PoliceReportFiled', 'Fault', 'WitnessPresent', 'AgentType', 'FraudFound']]
            cat_df = pd.concat([cat_dummies, a], axis=1)
            main_df = pd.concat([cat_df, num_cols], axis=1)

            logging.info("Data transformation completed")

            # input_feature_train_df=main_df.drop(columns=[target_column],axis=1)
            # target_feature_train_df=main_df[target_column]
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_data_path), exist_ok=True)
            main_df.to_csv(self.data_transformation_config.transformed_data_path, index=False, header=True)

            logging.info("train-test split initiated")
            train_set, test_set = train_test_split(main_df, test_size=0.2, random_state=15)
            
            train_set.to_csv(self.data_transformation_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_transformation_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed")

            return (self.data_transformation_config.train_data_path,
                    self.data_transformation_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)