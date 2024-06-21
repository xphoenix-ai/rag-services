import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings


class DocEmbeddings(Embeddings):
    def __init__(self):
        self.url = os.getenv("EMBED_URL")
    
    def __get_embeddings(self, sentences: List[str]) -> List[List[float]]:
        # if sentences:
        json_body = {
            "sentences": sentences
        }
        response = requests.post(self.url, json=json_body)
        response = response.json()
        embeddings = response["embeddings"]
        # else:
        #     embeddings = []
        
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        # print(f"texts: {texts} =====")
        return self.__get_embeddings(texts)
        
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        # print(f"texts: {text} -----")
        return self.__get_embeddings([text])[0]
