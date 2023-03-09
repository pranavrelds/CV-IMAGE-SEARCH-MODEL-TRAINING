from typing import List, Dict, Any
from pymongo import MongoClient

from src.entity.config_entity import DatabaseConfig

class MongoDBClient(object):
    def __init__(self):
        self.config = DatabaseConfig()
        url = self.config.URL.replace("<username>", self.config.USERNAME).replace("<password>", self.config.PASSWORD).replace("<cluster_name>", self.CLUSTER_NAME)
        self.client = MongoClient(url)

    def insert_bulk_record(self, documents: List[Dict[str, Any]]):
        try:
            db = self.client[self.config.DBNAME]
            collection = self.config.COLLECTION
            if collection not in db.list_collection_names():
                db.create_collection(collection)
            result = db[collection].insert_many(documents)
            return {"Response": "Success", "Inserted Documents": len(result.inserted_ids)}
        except Exception as e:
            raise e

    def get_collection_documents(self):
        try:
            db = self.client[self.config.DBNAME]
            collection = self.config.COLLECTION
            result = db[collection].find()
            return {"Response": "Success", "Info": result}
        except Exception as e:
            raise e

    def drop_collection(self):
        try:
            db = self.client[self.config.DBNAME]
            collection = self.config.COLLECTION
            db[collection].drop()
            return {"Response": "Success"}
        except Exception as e:
            raise e