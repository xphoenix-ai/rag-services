import os
import torch
import numpy as np
from typing import Tuple
from transformers import pipeline

from src.tts.tts_base import TTSBase


class TTS(TTSBase):
    """Hugging Face Transformers-based TTS implementation."""

    def __init__(self, model_path: str, device: str = "cpu", 
                 torch_dtype: torch.dtype = torch.float32, **model_kwargs) -> None:
        """Initialize Hugging Face TTS model.

        Args:
            model_path (str): Path to the TTS model
            device (str, optional): Device to load model on. Defaults to "cpu".
            torch_dtype (torch.dtype, optional): Data type for model weights. Defaults to torch.float32.
            **model_kwargs: Additional model configuration parameters
        """
        self.model = None
        super().__init__("hf", model_path, device, torch_dtype, **model_kwargs)

    def _load_model(self, model_path: str, device: str, torch_dtype: torch.dtype, 
                   **model_kwargs) -> None:
        """Load TTS model and pipeline.

        Args:
            model_path (str): Path to the TTS model
            device (str): Device to load model on
            torch_dtype (torch.dtype): Data type for model weights
            **model_kwargs: Additional model configuration parameters
        """
        self.model = pipeline(
            "text-to-speech", 
            model_path, 
            device=device,
            torch_dtype=torch_dtype,
            token=os.getenv("HF_TOKEN"),
            model_kwargs=model_kwargs
        )

        try:
            self.sr = self.model.model.generation_config.sample_rate
        except AttributeError:
            self.sr = self.model.model.config.sampling_rate

        print("[INFO] TTS service started...")

    def synthesize_one(self, text: str, **generation_config: dict) -> Tuple[int, np.ndarray]:
        """Synthesize audio from a single text input.

        Args:
            text (str): Text to synthesize
            **generation_config: Generation configuration parameters

        Returns:
            Tuple[int, np.ndarray]: Tuple containing (sample_rate, audio_array)
        """
        speech = self.model(text, **generation_config)
        return speech["sampling_rate"], speech["audio"].squeeze()
