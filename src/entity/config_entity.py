import os
from from_root import from_root

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

class ModelConfig:
    def __init__(self):
        self.LABEL = 101
        self.STORE_PATH = os.path.join(from_root(), "model", "benchmark")
        self.REPOSITORY = 'pytorch/vision:v0.10.0'
        self.BASEMODEL = 'resnet18'
        self.PRETRAINED = True

    def get_model_config(self):
        return self.__dict__

class TrainerConfig:
    def __init__(self):
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth")
        self.EPOCHS = 2
        self.Evaluation = True

    def get_trainer_config(self):
        return self.__dict__

class DatabaseConfig:
    def __init__(self):
        self.USERNAME: str = os.environ["DATABASE_USERNAME"]
        self.PASSWORD: str = os.environ["DATABASE_PASSWORD"]
        self.CLUSTER_NAME: str = os.environ["ATLAS_CLUSTER_PASSWORD"]
        self.URL: str = "mongodb+srv://<username>:<password>@<cluster_name>}.k6wb050.mongodb.net/?retryWrites=true&w=majority"
        self.DBNAME: str = "ImageSearch"
        self.COLLECTION: str = "Embeddings"

    def get_database_config(self):
        return self.__dict__

class ImageFolderConfig:
    def __init__(self):
        self.ROOT_DIR = os.path.join(from_root(), "data", "raw", "images")
        self.IMAGE_SIZE = 256
        self.LABEL_MAP = {}
        self.BUCKET: str = "image-database-system-01"
        self.S3_LINK = "https://{0}.s3.ap-south-1.amazonaws.com/images/{1}/{2}"

    def get_image_folder_config(self):
        return self.__dict__

class EmbeddingsConfig:
    def __init__(self):
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth")

    def get_embeddings_config(self):
        return self.__dict__

ImageRecord = namedtuple("ImageRecord", ["img", "label", "s3_link"])