import nltk
import threading
import numpy as np
import soundfile as sf
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Union, Any, Tuple


class TTSBase(ABC):
    """base class for tts
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
    def synthesize_one(self, text: str, **generation_config: dict) -> Tuple[int, np.ndarray]:
        """Synthesizes audio from the given short text

        Args:
            text (str): The input text to be synthesized

        Returns:
            Tuple: A tuple containing the sample rate and the generated audio array
        """
        raise NotImplementedError
    
    def synthesize(self, text: str, **generation_config: dict) -> Tuple[int, np.ndarray]:
        """Synthesizes audio from the given long-form text

        Args:
            text (str): The input text to be synthesized

        Returns:
            Tuple: A tuple containing the sample rate and the generated audio array
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, **generation_config)
            pieces += [audio_array, silence.copy()]

        return sample_rate, np.concatenate(pieces)
    
    def is_ready(self) -> bool:
        if self.model is not None:
            return True
        return False
    
    @staticmethod
    def save_audio(audio_array: np.ndarray, sample_rate: int, file_name: str):
        """dump given audio array to a file

        Args:
            audio_array (np.ndarray): Audio array to be dumped
            sample_rate (int): Sampling arate of the audio array
            file_name (str): Target file path
        """
        sf.write(file_name, audio_array, samplerate=sample_rate)
