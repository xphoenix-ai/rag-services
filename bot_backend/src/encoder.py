import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings


class DocEmbeddings(Embeddings):
    """Document embedding service that communicates with an embedding server via HTTP."""
    
    def __init__(self):
        """Initialize the DocEmbeddings service.
        
        Reads the embedding service URL from the EMBED_URL environment variable.
        """
        self.url = os.getenv("EMBED_URL")
    
    def __get_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """Get embeddings for a list of sentences from the embedding server.
        
        Args:
            sentences (List[str]): List of sentences to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            requests.RequestException: If the HTTP request to the server fails
        """
        json_body = {
            "sentences": sentences
        }
        response = requests.post(self.url, json=json_body)
        response = response.json()
        embeddings = response["embeddings"]
        
        return embeddings
    
    def is_ready(self):
        """Check if the embedding service is available and ready.
        
        Returns:
            bool: True if the encoder service is ready, False otherwise
        """
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["encoder"]
        except:
            status = False
            
        return status
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.
        
        Args:
            texts (List[str]): List of document texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.__get_embeddings(texts)
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        
        Args:
            text (str): Query text to embed
            
        Returns:
            List[float]: Embedding vector for the query text
        """
        return self.__get_embeddings([text])[0]
