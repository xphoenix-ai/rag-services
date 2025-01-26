import nltk
import threading
import numpy as np
import soundfile as sf
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Union, Any, Tuple

from utils.mapping.language_utils import get_language_code


class TTSBase(ABC):
    """Base class for Text-to-Speech models."""
    
    def __init__(self, class_name: str, *args: Any, **kwargs: Any) -> None:
        """Initialize the TTS base class.

        Args:
            class_name (str): Name of the TTS class
            *args (Any): Variable length argument list for model initialization
            **kwargs (Any): Arbitrary keyword arguments for model initialization
        """
        self.model = None
        self.sr = None
        self.class_name = class_name
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
    def synthesize_one(self, text: str, **generation_config: dict) -> Tuple[int, np.ndarray]:
        """Synthesize audio from a single text input.

        Args:
            text (str): Text to synthesize
            **generation_config: Generation configuration parameters

        Returns:
            Tuple[int, np.ndarray]: Tuple containing (sample_rate, audio_array)

        Raises:
            NotImplementedError: Must be implemented by child classes
        """
        raise NotImplementedError
    
    def synthesize(self, text: str, **generation_config: dict) -> Tuple[int, np.ndarray]:
        """Synthesize audio from long-form text by splitting into sentences.

        Args:
            text (str): Text to synthesize
            **generation_config: Generation configuration parameters

        Returns:
            Tuple[int, np.ndarray]: Tuple containing (sample_rate, audio_array)
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.sr))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize_one(sent, **generation_config)
            pieces += [audio_array, silence.copy()]

        return sample_rate, np.concatenate(pieces)

    def get_lang_code(self, lang_name: str) -> Tuple[str, str]:
        """Get standardized language code for given language name.

        Args:
            lang_name (str): Name or code of the language

        Returns:
            Tuple[str, str]: Tuple containing (language_code, error_message)
        """
        return get_language_code("tts", self.class_name, lang_name)
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference.

        Returns:
            bool: True if model is loaded and ready, False otherwise
        """
        if self.model is not None:
            return True
        return False
    
    @staticmethod
    def save_audio(audio_array: np.ndarray, sample_rate: int, file_name: str) -> None:
        """Save audio array to file.

        Args:
            audio_array (np.ndarray): Audio data to save
            sample_rate (int): Sample rate of the audio data
            file_name (str): Path to save the audio file
        """
        sf.write(file_name, audio_array, samplerate=sample_rate)

    @staticmethod
    def play_audio(audio_array: np.ndarray, sample_rate: int) -> None:
        """Play audio array through system audio.

        Args:
            audio_array (np.ndarray): Audio data to play
            sample_rate (int): Sample rate of the audio data
        """
        sf.play(audio_array, sample_rate)
        sf.wait()