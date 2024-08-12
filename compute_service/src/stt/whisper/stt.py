import whisper
import numpy as np
from typing import Tuple

from src.stt.stt_base import STTBase


class STT(STTBase):
    def __init__(self, model_path: str, device: str="cpu", **model_kwargs):
        self.model = None
        super().__init__(model_path, device, **model_kwargs)

    def _load_model(self, model_path: str, device: str, **model_kwargs) -> None:
        self.model = whisper.load_model(model_path, device=device)

    def transcribe(self, audio_array: np.ndarray, **generation_config: dict) -> str:
        transcription = self.model.transcribe(audio_array, **generation_config)
        
        return transcription["text"].strip()
