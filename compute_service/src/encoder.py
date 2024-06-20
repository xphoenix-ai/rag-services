import torch
from sentence_transformers import SentenceTransformer


class Encoder:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path).to(self.device)
        
    def encode(self, sentences: list):
        torch.cuda.empty_cache()
        embeddings = self.model.encode(sentences)
        torch.cuda.empty_cache()
        
        return embeddings
