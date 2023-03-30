import os 
import sys
import pandas as pd
from dataclasses import dataclass
import joblib
import argparse

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.components import model_dispatcher
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig

from sklearn.metrics import classification_report, f1_score

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join('artifacts\models', 'model.pkl')

class train_specific_model():
    def __init__(self):
        self.model_train_config = ModelTrainConfig()
    
    def run_specific_model(self, train_data_path, test_data_path, model_name):
        try:
            os.makedirs(os.path.dirname(self.model_train_config.trained_model_file_path), exist_ok=True)
            logging.info(f"Reading train and test data for {model_name}")
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info('split train-test input data')
            X_train, y_train, X_test, y_test = (train_data.drop('FraudFound',axis=1),
                                                train_data['FraudFound'].values,
                                                test_data.drop('FraudFound',axis=1),
                                                test_data['FraudFound'].values)

            #modelling
            logging.info('Getting specified model')
            clf = model_dispatcher.specific_models[model_name]
            clf.fit(X_train, y_train)

            logging.info('Model fit complete')

            train_f1score = f1_score(clf.predict(X_train),y_train, average='weighted')
            test_f1score = f1_score(clf.predict(X_test),y_test, average='weighted')

            pred = clf.predict(X_test)
            pred_f1score = f1_score(pred, y_test, average='weighted')
            print(classification_report(y_test, pred))
            logging.info('Model prediction complete')
        
            #save the model
            logging.info('Saving the specific model')
            joblib.dump(clf, 
                        os.path.join('artifacts\models', f"{model_name}.bin"))
            
            return pred_f1score
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
     #initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    parser.add_argument(
                "--model",
                type=str
                )
    # parser.add_argument(
    #             "--criterion",
    #             type=str
    #             )
                    
    # read the arguments from the command line
    args = parser.parse_args()

# run the fold specified by command line arguments
    obj=DataIngestion()
    raw_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_data, test_data = data_transformation.initiate_data_transformation(raw_data)

    train = train_specific_model()
    print(train.run_specific_model(train_data_path=train_data, 
                             test_data_path=test_data, 
                             model_name=args.model))