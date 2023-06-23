import os
import sys
import numpy as np
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging

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