import whisper
import numpy as np
from typing import Tuple

from src.stt.stt_base import STTBase


class STT(STTBase):
    """OpenAI Whisper-based STT implementation."""

    def __init__(self, language: str, model_path: str, device: str = "cpu", **model_kwargs) -> None:
        """Initialize OpenAI Whisper STT model.

        Args:
            language (str): Default language for transcription
            model_path (str): Name/size of the Whisper model (e.g., "base", "small", "medium", "large")
            device (str, optional): Device to load model on. Defaults to "cpu".
            **model_kwargs: Additional model configuration parameters
        """
        self.model = None
        super().__init__("whisper", language, model_path, device, **model_kwargs)

    def _load_model(self, model_path: str, device: str, **model_kwargs) -> None:
        """Load Whisper model.

        Args:
            model_path (str): Name/size of the Whisper model
            device (str): Device to load model on
            **model_kwargs: Additional model configuration parameters
        """
        self.model = whisper.load_model(model_path, device=device)
        print("[INFO] STT service started...")

    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language: str, 
                  **generation_config: dict) -> Tuple[str, str]:
        """Transcribe audio using OpenAI Whisper model.

        Args:
            audio_array (np.ndarray): The audio data to transcribe
            sample_rate (int): Sample rate of the audio data
            language (str): Language of the audio
            **generation_config: Additional configuration for transcription

        Returns:
            Tuple[str, str]: Tuple containing (transcribed_text, language)
        """
        lang_code, lang_error = self.get_lang_code(language)
        transcription = self.model.transcribe(audio_array, language=lang_code, **generation_config)
        
        return transcription["text"].strip(), language
