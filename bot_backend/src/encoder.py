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
        
        return response["embeddings"]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.__get_embeddings(texts)
        
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.__get_embeddings([text])
