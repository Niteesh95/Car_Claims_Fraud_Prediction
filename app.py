import sys
import pandas as pd
import numpy as np
from src.exception import CustomException

X = ['MaritalStatus_Divorced', 'MaritalStatus_Married',
       'MaritalStatus_Single', 'MaritalStatus_Widow',
       'PolicyType_Sedan - All Perils', 'PolicyType_Sedan - Collision',
       'PolicyType_Sedan - Liability', 'PolicyType_Sport - All Perils',
       'PolicyType_Sport - Collision', 'PolicyType_Sport - Liability',
       'PolicyType_Utility - All Perils', 'PolicyType_Utility - Collision',
       'PolicyType_Utility - Liability', 'VehiclePrice_20,000 to 29,000',
       'VehiclePrice_30,000 to 39,000', 'VehiclePrice_40,000 to 59,000',
       'VehiclePrice_60,000 to 69,000', 'VehiclePrice_less than 20,000',
       'VehiclePrice_more than 69,000', 'Days:Policy-Accident_1 to 7',
       'Days:Policy-Accident_15 to 30', 'Days:Policy-Accident_8 to 15',
       'Days:Policy-Accident_more than 30', 'Days:Policy-Accident_none',
       'Days:Policy-Claim_15 to 30', 'Days:Policy-Claim_8 to 15',
       'Days:Policy-Claim_more than 30', 'Days:Policy-Claim_none',
       'AgeOfVehicle_2 years', 'AgeOfVehicle_3 years', 'AgeOfVehicle_4 years',
       'AgeOfVehicle_5 years', 'AgeOfVehicle_6 years', 'AgeOfVehicle_7 years',
       'AgeOfVehicle_more than 7', 'AgeOfVehicle_new',
       'NumberOfSuppliments_1 to 2', 'NumberOfSuppliments_3 to 5',
       'NumberOfSuppliments_more than 5', 'NumberOfSuppliments_none',
       'AddressChange-Claim_1 year', 'AddressChange-Claim_2 to 3 years',
       'AddressChange-Claim_4 to 8 years', 'AddressChange-Claim_no change',
       'AddressChange-Claim_under 6 months', 'NumberOfCars_1 vehicle',
       'NumberOfCars_2 vehicles', 'NumberOfCars_3 to 4', 'NumberOfCars_5 to 8',
       'NumberOfCars_more than 8', 'Age_Group_0-9', 'Age_Group_10-19',
       'Age_Group_20-29', 'Age_Group_30-39', 'Age_Group_40-49',
       'Age_Group_50-59', 'Age_Group_60-69', 'Age_Group_70-79',
       'Age_Group_80-89', 'AccidentArea', 'Sex', 'PoliceReportFiled', 'Fault',
       'WitnessPresent', 'AgentType', 'Age', 'Deductible']

def predict_fraud(model, age, deductible, MaritalStatus, PolicyType, VehiclePrice,
                  daysAccident, daysclaim, AgeOfVehicle, NumberOfSuppliments, addrchangeclaim, 
                  NumberOfCars, Age_Group, AccidentArea, Sex, PoliceReportFiled, Fault, 
                  WitnessPresent, AgentType):    
    try:
        marital_status = X.index('MaritalStatus_' + MaritalStatus)
        policy_type = X.index('PolicyType_' + PolicyType)
        vehicle_price = X.index('VehiclePrice_' + VehiclePrice)
        days_policy_accident = X.index('Days:Policy-Accident_' + daysAccident)
        days_policy_claim = X.index('Days:Policy-Claim_' + daysclaim)
        vehicle_age = X.index('AgeOfVehicle_' + AgeOfVehicle)
        no_of_suppliments = X.index('NumberOfSuppliments_' + NumberOfSuppliments)
        addr_change_claim = X.index('AddressChange-Claim_' + addrchangeclaim)
        no_of_cars = X.index('NumberOfCars_' + NumberOfCars)
        age_group = X.index('Age_Group_' + Age_Group)

        index_list = [ marital_status, policy_type, vehicle_price, days_policy_accident, days_policy_claim,
                      vehicle_age, no_of_suppliments,addr_change_claim, no_of_cars,age_group ]

        x = np.zeros(len(X))
        x[0] = age
        x[1] = deductible
        x[2] = AccidentArea
        x[3] = Sex
        x[4] = PoliceReportFiled
        x[5] = Fault
        x[6] = WitnessPresent
        x[7] = AgentType

        for ind in index_list:
            if ind >= 0:
                x[ind] = 1


        return model.predict([x])[0]
    except Exception as e:
        raise CustomException(e, sys)