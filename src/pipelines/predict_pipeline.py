import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        age: str,
        deductible: str,
        accidentarea: str,
        sex: str,
        PoliceReportFiled: str,
        fault: str,
        witnesspresent: str,
        agenttype: str,
        maritalstatus: str,
        policytype: str,
        vehicleprice: str,
        DaysSinceAccident: str,
        DaysSinceClaim: str,
        ageofvehicle: str,
        noofsuppliments: str,
        addresschangeclaim: str,
        noofcars: str):

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)