import os
import torch
import numpy as np
from typing import Tuple
from transformers import pipeline

from src.tts.tts_base import TTSBase


class TTS(TTSBase):
    def __init__(self, model_path: str, device: str="cpu", torch_dtype: torch.dtype=torch.float32, **model_kwargs):
        self.model = None
        super().__init__(model_path, device, torch_dtype, **model_kwargs)

    def _load_model(self, model_path: str, device: str, torch_dtype: torch.dtype, **model_kwargs) -> None:
        self.model = pipeline(
            "text-to-speech", 
            model_path, 
            device=device,
            torch_dtype=torch_dtype,
            token=os.getenv("HF_TOKEN"),
            model_kwargs=model_kwargs
        )
        self.sr = self.model.model.generation_config.sample_rate
        print("[INFO] TTS service started...")

    def synthesize_one(self, text: str, **generation_config: dict) -> Tuple[int, np.ndarray]:
        speech = self.model(text, **generation_config)

        return speech["sampling_rate"], speech["audio"]
