import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import (confusion_matrix,accuracy_score, 
                            classification_report, ConfusionMatrixDisplay, f1_score)

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train,X_test,y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_cm = confusion_matrix(y_train, y_train_pred)
            train_cm_show = ConfusionMatrixDisplay(confusion_matrix = train_cm, 
                                                   display_labels = model.classes_)

            test_cm = confusion_matrix(y_test, y_test_pred)
            test_cm_show = ConfusionMatrixDisplay(confusion_matrix = test_cm, 
                                                  display_labels = model.classes_)
            
            train_cr = classification_report(y_train, y_train_pred)
            test_cr = classification_report(y_test, y_test_pred)

            train_f1score = f1_score(y_train_pred,y_train, average='weighted')
            test_f1score = f1_score(y_test_pred,y_test, average='weighted')

            report[list(models.keys())[i]] = test_f1score
            # report[list(models.keys())[i]] = test_model_score

            return report
    except Exception as e:
        raise CustomException(e, sys)
