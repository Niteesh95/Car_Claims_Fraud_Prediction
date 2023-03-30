import os 
import sys
import pandas as pd
from dataclasses import dataclass
import joblib

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.components import model_dispatcher, model_hyperparameters
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig

from sklearn.metrics import classification_report, f1_score

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join('artifacts\models', 'model.pkl')

class train_model():
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

    def initiateModelTrain(self, train_data_path, test_data_path):
        try:
            os.makedirs(os.path.dirname(self.model_train_config.trained_model_file_path), exist_ok=True)
            logging.info("Reading train and test data")
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info('split train-test input data')
            X_train, y_train, X_test, y_test = (train_data.drop('FraudFound',axis=1),
                                                train_data['FraudFound'].values,
                                                test_data.drop('FraudFound',axis=1),
                                                test_data['FraudFound'].values)

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train,
                                             X_test=X_test,y_test=y_test, 
                                             models=model_dispatcher.models,
                                             params = model_hyperparameters.parameters)
            
            # print(model_report)
            
            #get best model
            best_f1score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_f1score)]

            best_model = model_dispatcher.models[best_model_name]

            logging.info('Best model found')

            pred = best_model.predict(X_test)
            pred_f1score = f1_score(pred, y_test, average='weighted')

            logging.info('Saving the best model')
            # print(best_model)
            joblib.dump(best_model, 
                        os.path.join('artifacts\models', f"bestmodel_{best_model_name}.bin"))

            return pred_f1score
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    raw_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_data, test_data = data_transformation.initiate_data_transformation(raw_data)

    model_trainer = train_model()
    print(model_trainer.initiateModelTrain(train_data,test_data))