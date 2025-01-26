import threading
import numpy as np
from typing import Tuple, Any
from abc import ABC, abstractmethod

from utils.mapping.language_utils import get_language_code


class STTBase(ABC):
    """Base class for Speech-to-Text models."""
    
    def __init__(self, class_name: str, language: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the STT base class.

        Args:
            class_name (str): Name of the STT class
            language (str): Default language for transcription
            *args (Any): Variable length argument list for model initialization
            **kwargs (Any): Arbitrary keyword arguments for model initialization
        """
        self.model = None
        self.class_name = class_name
        self.language = language
        self.init(*args, **kwargs)
        
    def init(self, *args: Any, **kwargs: Any) -> None:
        """Initialize model loading in background thread.

        Args:
            *args (Any): Variable length argument list passed to _load_model
            **kwargs (Any): Arbitrary keyword arguments passed to _load_model
        """
        thread = threading.Thread(target=self._load_model, args=(args), kwargs=kwargs)
        thread.start()
        
    @abstractmethod
    def _load_model(self, *args: Any, **kwargs: Any) -> None:
        """Initialize models - to be implemented by child classes.

        Args:
            *args (Any): Variable length argument list for model initialization
            **kwargs (Any): Arbitrary keyword arguments for model initialization

        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError
                
    @abstractmethod
    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language: str, 
                  **generation_config: dict) -> Tuple[str, str]:
        """Transcribe audio to text.

        Args:
            audio_array (np.ndarray): The audio data to transcribe
            sample_rate (int): Sample rate of the audio data
            language (str): Language of the audio
            **generation_config: Additional configuration for transcription

        Returns:
            Tuple[str, str]: Tuple containing (transcribed_text, error_message)

        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError

    def get_lang_code(self, lang_name: str) -> Tuple[str, str]:
        """Get standardized language code for given language name.

        Args:
            lang_name (str): Name or code of the language

        Returns:
            Tuple[str, str]: Tuple containing (language_code, error_message)
        """
        return get_language_code("stt", self.class_name, lang_name)

    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference.

        Returns:
            bool: True if model is loaded and ready, False otherwise
        """
        if self.model is not None:
            return True
        return False
