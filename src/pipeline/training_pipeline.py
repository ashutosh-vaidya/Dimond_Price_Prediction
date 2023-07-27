
from src.components.data_ingetion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# The code block `if __name__=='__main__':` is a common Python idiom that allows a script to be
# executed as a standalone program.
if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.init_data_ingestion()
    data_transfromation = DataTransformation()
    train_arr, test_arr, _ = data_transfromation.init_data_transformation(
        train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.init_model_training(train_arr, test_arr)
