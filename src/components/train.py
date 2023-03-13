import os 
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score,KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score

@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class train_model():
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

    def initiateModelTrain(self, train_data_path, test_data_path):
        try:
            logging.info("Reading train and test data")
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            logging.info('split train-test input data')
            X_train, y_train, X_test, y_test = (train_data.drop('FraudFound',axis=1),
                                                train_data['FraudFound'].values,
                                                test_data.drop('FraudFound',axis=1),
                                                test_data['FraudFound'].values)
            
            models = {
            "random_forest_gini": RandomForestClassifier(criterion='gini', 
                                                         max_depth=5,  
                                                         n_estimators=6, 
                                                         n_jobs=3),
                
            "xgboost_gbtree": XGBClassifier(learning_rate=0.1, 
                                            max_depth=4, 
                                            booster='gbtree'),

            "decision_tree_gini": DecisionTreeClassifier(criterion='gini', 
                                                            max_depth=5, 
                                                            max_features=6, 
                                                            random_state=15)
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train,
                                             X_test=X_test,y_test=y_test, 
                                             models=models)
            
            #get best model
            best_f1score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_f1score)]

            best_model = models[best_model_name]

            logging.info('Best model found')

            pred = best_model.predict(X_test)
            pred_f1score = f1_score(pred, y_test, average='weighted')

            return pred_f1score
        except Exception as e:
            raise CustomException(e, sys)