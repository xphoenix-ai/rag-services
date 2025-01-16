import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Union, Any


class EncoderBase(ABC):
    """base class for encoders
    """
    def __init__(self, class_name, *args, **kwargs):
        self.model = None
        self.class_name = class_name
        self.init(*args, **kwargs)
        
    def init(self, *args, **kwargs) -> None:
        """
        model loading is done in background
        """
        thread = threading.Thread(target=self._load_model, args=(args), kwargs=kwargs)
        thread.start()
        
    @abstractmethod
    def _load_model(self, *args, **kwargs) -> None:
        """initialize models
        """
        raise NotImplementedError
        
    @abstractmethod
    def encode(self, sentences: List[str]) -> List[List[float]]:
        """return embeddings from a given sequence of texts

        Args:
            sentences (List[str]): list of texts

        Returns:
            List[List[float]]: list of embedding vectors
        """
        raise NotImplementedError
    
    def is_ready(self) -> bool:
        if self.model is not None:
            return True
        return False
