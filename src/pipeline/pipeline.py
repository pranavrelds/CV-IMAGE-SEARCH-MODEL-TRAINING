import os
import torch
from tqdm import tqdm
from from_root import from_root
from torch.utils.data import DataLoader

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model import NeuralNet

class Pipeline:
    def __init__(self):
        self.paths = ["data", "data/raw", "data/splitted", "data/embeddings",
                      "model", "model/benchmark", "model/finetuned"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def initiate_data_ingestion(self):
        for folder in self.paths:
            path = os.path.join(from_root(), folder)
            if not os.path.exists(path):
                os.mkdir(folder)

        data_ingestion = DataIngestion()
        data_ingestion.run_step()     

    @staticmethod
    def initiate_data_preprocessing():
        dp = DataPreprocessing()
        loaders = dp.run_step()
        return loaders   

    @staticmethod
    def initiate_model_architecture():
        return NeuralNet()

    def run_pipeline(self):
        self.initiate_data_ingestion()
        loaders = self.initiate_data_preprocessing()
        net = self.initiate_model_architecture()
        return {"Response": "Pipeline Run Complete"}


if __name__ == "__main__":
    image_search_pipeline = Pipeline()
    image_search_pipeline.run_pipeline()