import os
import sys
import numpy as np
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        #Path to directory to save the file
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        #Create a file and save it
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception occured while save_object.")
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        logging.info("Model Evaluation started...")
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            #train the model
            model.fit(X_train, y_train)

            #predict the value
            y_test_prod = model.predict(X_test)

            #Get R2 scores for train and test data. (We can also use Adjusted R2)
            test_model_score = r2_score(y_test, y_test_prod)

            report[list(models.keys())[i]] = test_model_score

        return report
        logging.info("Model Evaluation completed.")

    except Exception as e:
        logging.info("Exception occured during evalute_model")
        raise CustomException(e, sys)