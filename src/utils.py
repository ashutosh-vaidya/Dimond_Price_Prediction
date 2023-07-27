import os
import sys
import numpy as np
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    """
    The function `save_object` saves an object to a file using pickle in Python, creating the necessary
    directory if it doesn't exist.

    Parameters:

    file_path (str) -- The file path where the object will be saved. This should include the file name and extension

    obj -- The object that you want to save to a file. It can be any Python object that is serializable.
    """
    try:
        # Path to directory to save the file
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        # Create a file and save it
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info("Exception occured while save_object.")
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    The function `evaluate_model` trains multiple models on the training data, predicts the values for
    the test data, and returns a report of the R2 scores for each model.
    """
    try:
        logging.info("Model Evaluation started...")
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # train the model
            model.fit(X_train, y_train)

            # predict the value
            y_test_prod = model.predict(X_test)

            # Get R2 scores for train and test data. (We can also use Adjusted R2)
            test_model_score = r2_score(y_test, y_test_prod)

            report[list(models.keys())[i]] = test_model_score

        logging.info("Model Evaluation completed.")

        return report

    except Exception as e:
        logging.info("Exception occured during evalute_model")
        raise CustomException(e, sys)


def load_object(file_path):
    """
    The function `load_object` loads an object from a file using the `pickle` module in Python, and logs
    any exceptions that occur.

    Parameter:

    file_path -- The file path is the location of the file that you want to load. It should be a
    string that specifies the path to the file, including the file name and extension. For example,
    "C:/Users/username/Documents/myfile.pkl" or "data/myfile.pkl"
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured while loading a file")
        raise CustomException(e, sys)
