import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings


class DocEmbeddings(Embeddings):
    def __init__(self):
        self.url = os.getenv("EMBED_URL")
    
    def __get_embeddings(self, sentences: List[str]) -> List[List[float]]:
        json_body = {
            "sentences": sentences
        }
        response = requests.post(self.url, json=json_body)
        response = response.json()
        embeddings = response["embeddings"]
        
        return embeddings
    
    def is_ready(self):
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["encoder"]
        except:
            status = False
            
        return status
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.__get_embeddings(texts)
        
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.__get_embeddings([text])[0]
