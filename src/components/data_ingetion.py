import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

# Initialize the data ingetion config
@dataclass
class DataIngestionConfig:
    #Create a path for saving the train, test files after splitting the dataset
    #raw file will contain original unsplit file
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')
    raw_data_path : str = os.path.join('artifacts', 'raw.csv')

# Class for data ingestion
class DataIngestion:
    #constructor
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("Data Ingestion has started...")
        try:
            #reading the data from the file and saving it into the dataframe
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info("Reading of the data into the pandas dataframe completed.")

            #save the data into the raw file before applying train test split
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            #Split the data into train test data
            logging.info("Splitting data into train test set")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)            

            logging.info("Data Ingestion is completed.")
            #return the train test data path
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured at the Data Ingesgtion stage.")
            raise CustomException(e, sys)

    