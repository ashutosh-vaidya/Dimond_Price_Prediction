
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        """
        Loads pre-trained models, scales the input features, and makes predictions using the loaded model.
        """
        try:
            logging.info("Prediction started...")
            # load models
            # preprocessor_path = "artifacts\preprocessor.pkl" #This works only for windows
            preprocessor_path = os.path.join(
                "artifacts", "preprocessor.pkl")  # Platform independent
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # scale the data
            data_scaled = preprocessor.transform(features)

            # prediction
            pred = model.predict(data_scaled)
            logging.info("Prediction Completed.")
            return pred
        except Exception as e:
            logging.info("Exception occurred in Prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, carat: float, depth: float, table: float, x: float, y: float, z: float,
                 cut: str, color: str, clarity: str):

        # initialization
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        """
        Converts input data into a DataFrame object in Python.
        """
        try:
            logging.info("Converting input data to Dataframe")
            custom_data_input_dict = {
                "carat": [self.carat],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe generated")
            return df
        except Exception as e:
            logging.info(
                "Exception occurred during get_data_as_dataframe method")
            raise CustomException(e, sys)
