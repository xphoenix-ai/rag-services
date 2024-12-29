import os
import torch
import librosa
import numpy as np
from typing import Tuple
from transformers import pipeline, WhisperProcessor

from src.stt.stt_base import STTBase


class STT(STTBase):
    def __init__(self, language, model_path: str, device: str="cpu", torch_dtype: torch.dtype=torch.float32, **model_kwargs):
        self.model = None
        super().__init__(language, model_path, device, torch_dtype, **model_kwargs)

    def _load_model(self, model_path: str, device: str, torch_dtype: torch.dtype, **model_kwargs) -> None:
        torch_dtype = torch.float32 if device == "cpu" else torch_dtype
        self.model = pipeline(
            "automatic-speech-recognition", 
            model_path, 
            device=device,
            torch_dtype=torch_dtype,
            token=os.getenv("HF_TOKEN"),
            **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe")
        print("[INFO] STT service started...")

    def transcribe(self, audio_array: np.ndarray, sample_rate: int, **generation_config: dict) -> Tuple[str, str]:
        # Ensure audio is mono channel
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz
        if sample_rate != 16000:
            print(f"[INFO] resampling {sample_rate}Hz --> 16000Hz")
            audio_array = librosa.resample(
                y=audio_array,
                orig_sr=sample_rate,
                target_sr=16000
            )

        # Convert to float32 numpy array
        audio_array = audio_array.astype(np.float32)

        # Normalize audio
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()

        generate_kwargs = generation_config.pop("generate_kwargs")
        generate_kwargs["forced_decoder_ids"] = self.forced_decoder_ids
        transcription = self.model(
            {"sampling_rate": 16_000, "raw": audio_array},
            **generation_config,
            generate_kwargs=generate_kwargs
        )

        return transcription["text"].strip(), self.language
