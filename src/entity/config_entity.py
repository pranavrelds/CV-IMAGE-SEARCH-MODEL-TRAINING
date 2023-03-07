class DataIngestionConfig:
    def __init__(self):
        self.PREFIX: str = "images/"
        self.RAW: str = "data/raw"
        self.SPLIT: str = "data/splitted"
        self.BUCKET: str = "image-database-system-01"
        self.SEED: int = 1337
        self.RATIO: tuple = (0.8, 0.1, 0.1)

    def get_data_ingestion_config(self):
        return self.__dict__

class DataPreprocessingConfig:
    def __init__(self):
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 256
        self.TRAIN_DATA_PATH = os.path.join(from_root(), "data", "splitted", "train")
        self.TEST_DATA_PATH = os.path.join(from_root(), "data", "splitted", "test")
        self.VALID_DATA_PATH = os.path.join(from_root(), "data", "splitted", "valid")

    def get_data_preprocessing_config(self):
        return self.__dict__