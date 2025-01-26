import torch
from typing import List
from sentence_transformers import SentenceTransformer

from src.encoder.encoder_base import EncoderBase


class Encoder(EncoderBase):
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        """Initialize the Sentence Transformer encoder.

        Args:
            model_path (str): Path to the sentence transformer model
            device (str, optional): Device to load the model on. Defaults to "cuda".
                Falls back to "cpu" if CUDA is not available.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        super().__init__("st", model_path)
        
    def _load_model(self, model_path: str) -> None:
        """Load the Sentence Transformer model.

        Args:
            model_path (str): Path to the sentence transformer model
        """
        self.model = SentenceTransformer(model_path).to(self.device)
        print("[INFO] Encoding service started...")
        
    def encode(self, sentences: List[str]) -> List[List[float]]:
        """Encode the input sentences into embeddings.

        Args:
            sentences (list): List of input texts to encode

        Returns:
            list: List of embedding vectors as float lists
        """
        torch.cuda.empty_cache()
        embeddings = self.model.encode(sentences, convert_to_numpy=True).tolist()
        torch.cuda.empty_cache()
        
        return embeddings
