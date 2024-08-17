import os
import torch
import numpy as np
from typing import Tuple
from transformers import pipeline

from src.stt.stt_base import STTBase


class STT(STTBase):
    def __init__(self, language, model_path: str, device: str="cpu", torch_dtype: torch.dtype=torch.float32, **model_kwargs):
        self.model = None
        super().__init__(language, model_path, device, torch_dtype, **model_kwargs)

    def _load_model(self, model_path: str, device: str, torch_dtype: torch.dtype, **model_kwargs) -> None:
        self.model = pipeline(
            "automatic-speech-recognition", 
            model_path, 
            device=device,
            torch_dtype=torch_dtype,
            token=os.getenv("HF_TOKEN"),
            model_kwargs=model_kwargs
        )
        print("[INFO] STT service started...")

    def transcribe(self, audio_array: np.ndarray, **generation_config: dict) -> str:
        transcription = self.model(audio_array, **generation_config)

        return transcription["text"].strip(), self.language
