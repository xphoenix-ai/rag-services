import threading
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class STTBase(ABC):
    """base class for stt
    """
    def __init__(self, language, *args, **kwargs):
        self.model = None
        self.language = language
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
    def transcribe(self, audio_array: np.ndarray, sample_rate:int, **generation_config: dict) -> Tuple[str, str]:
        """Transcribes the given audio data

        Args:
            audio_array (np.ndarray): The audio data to be transcribed
            sample_rate (int): Sample rate of audio_array
        Returns:
            Tuple[str, str]: The transcribed text and the language
        """
        raise NotImplementedError
    
    def is_ready(self) -> bool:
        if self.model is not None:
            return True
        return False
