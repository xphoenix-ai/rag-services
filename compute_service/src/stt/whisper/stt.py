import whisper
import numpy as np
from typing import Tuple

from src.stt.stt_base import STTBase


class STT(STTBase):
    def __init__(self, language: str, model_path: str, device: str="cpu", **model_kwargs):
        self.model = None
        super().__init__(language, model_path, device, **model_kwargs)

    def _load_model(self, model_path: str, device: str, **model_kwargs) -> None:
        self.model = whisper.load_model(model_path, device=device)
        print("[INFO] STT service started...")

    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language: str, **generation_config: dict) -> Tuple[str, str]:
        transcription = self.model.transcribe(audio_array, language=language, **generation_config)
        
        return transcription["text"].strip(), language
