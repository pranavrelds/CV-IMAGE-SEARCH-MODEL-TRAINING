import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from collections import namedtuple

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from src.components.model import NeuralNet
from src.components.data_preprocessing import DataPreprocessing
from src.entity.config_entity import ImageFolderConfig, EmbeddingsConfig, ImageRecord
from src.data_access.mongodb_operations import MongoDBClient


class ImageFolder(Dataset):
    def __init__(self, label_map: Dict):
        self.config = ImageFolderConfig()
        self.config.LABEL_MAP = label_map
        self.transform = self.transformations()
        self.image_records: List[ImageRecord] = []
        self.record = ImageRecord

        file_list = os.listdir(self.config.ROOT_DIR)

        for class_path in file_list:
            path = os.path.join(self.config.ROOT_DIR, f"{class_path}")
            images = os.listdir(path)
            for image in tqdm(images):
                image_path = Path(f"""{self.config.ROOT_DIR}/{class_path}/{image}""")
                self.image_records.append(self.record(img=image_path,
                                                      label=self.config.LABEL_MAP[class_path],
                                                      s3_link=self.config.S3_LINK.format(self.config.BUCKET, class_path,
                                                                                         image)))

    def transformations(self):
        TRANSFORM_IMG = transforms.Compose(
            [transforms.Resize(self.config.IMAGE_SIZE),
             transforms.CenterCrop(self.config.IMAGE_SIZE),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]
        )

        return TRANSFORM_IMG

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, idx):
        record = self.image_records[idx]
        images, targets, links = record.img, record.label, record.s3_link
        images = Image.open(images)

        if len(images.getbands()) < 3:
            images = images.convert('RGB')
        images = np.array(self.transform(images))
        targets = torch.from_numpy(np.array(targets))
        images = torch.from_numpy(images)

        return images, targets, links


class EmbeddingGenerator:
    def __init__(self, model, device):
        self.config = EmbeddingsConfig()
        self.mongo = MongoDBClient()
        self.model = model
        self.device = device
        self.embedding_model = self.load_model()
        self.embedding_model.eval()

    def load_model(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.config.MODEL_STORE_PATH, map_location=self.device))
        return nn.Sequential(*list(model.children())[:-1])

    def run_step(self, batch_size, image, label, s3_link):
        records = dict()

        images = self.embedding_model(image.to(self.device))
        images = images.detach().cpu().numpy()

        records['images'] = images.tolist()
        records['label'] = label.tolist()
        records['s3_link'] = s3_link

        df = pd.DataFrame(records)
        records = list(json.loads(df.T.to_json()).values())
        self.mongo.insert_bulk_record(records)

        return {"Response": f"Completed Embeddings Generation for {batch_size}."}


if __name__ == "__main__":
    dp = DataPreprocessing()
    loaders = dp.run_step()

    data = ImageFolder(label_map=loaders["valid_data_loader"][1].class_to_idx)
    dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)
    embeds = EmbeddingGenerator(model=NeuralNet(), device="cpu")

    for batch, values in tqdm(enumerate(dataloader)):
        img, target, link = values
        print(embeds.run_step(batch, img, target, link))


