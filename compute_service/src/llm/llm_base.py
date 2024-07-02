import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Union, Any


class LLMBase(ABC):
    """base class for llms
    """
    def __init__(self, *args, **kwargs):
        self.model = None
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
    def generate(self, prompt: str, **generation_config: dict) -> str:
        """get llm output

        Args:
            prompt (str): prompt for the llm

        Returns:
            str: llm response
        """
        raise NotImplementedError
    
    def is_ready(self) -> bool:
        if self.model is not None:
            return True
        return False
