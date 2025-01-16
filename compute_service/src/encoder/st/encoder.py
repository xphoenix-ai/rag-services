import torch
from sentence_transformers import SentenceTransformer

from src.encoder.encoder_base import EncoderBase


class Encoder(EncoderBase):
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        super().__init__("st", model_path)
        
    def _load_model(self, model_path):
        self.model = SentenceTransformer(model_path).to(self.device)
        print("[INFO] Encoding service started...")
        
    def encode(self, sentences: list):
        torch.cuda.empty_cache()
        embeddings = self.model.encode(sentences, convert_to_numpy=True).tolist()
        torch.cuda.empty_cache()
        
        return embeddings
