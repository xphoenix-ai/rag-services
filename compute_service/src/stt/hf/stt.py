import os
import torch
import librosa
import numpy as np
from typing import Tuple
from transformers import pipeline, WhisperConfig, WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig

from src.stt.stt_base import STTBase


class STT(STTBase):
    def __init__(self, language, model_path: str, device: str = "cpu", torch_dtype: torch.dtype = torch.float32,
                 **model_kwargs):
        self.model = None
        super().__init__("hf", language, model_path, device, torch_dtype, **model_kwargs)

    def __get_base_model(self, model_name):
        if 'large-v3' in model_name:
            base_model = 'openai/whisper-large-v3'
        elif 'large-v2' in model_name:
            base_model = 'openai/whisper-large-v2'
        elif 'medium' in model_name:
            base_model = 'openai/whisper-medium'
        elif 'small' in model_name:
            base_model = 'openai/whisper-small'
        elif 'base' in model_name:
            base_model = 'openai/whisper-base'
        elif 'tiny' in model_name:
            base_model = 'openai/whisper-tiny'
        else:
            raise Exception("base model could not be found! model_name should contain that.")

        return base_model

    def _load_model(self, model_path: str, device: str, torch_dtype: torch.dtype, **model_kwargs) -> None:
        base_model = self.__get_base_model(model_path)
        torch_dtype = torch.float32 if device == "cpu" else torch_dtype
        self.hf_model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
            token=os.getenv("HF_TOKEN"),
            **model_kwargs
        )
        self.config = WhisperConfig.from_pretrained(base_model)
        self.hf_model.config = self.config
        self.processor = WhisperProcessor.from_pretrained(base_model, language=self.language)
        self.base_generation_config = GenerationConfig.from_pretrained(base_model)
        self.hf_model.generation_config = self.base_generation_config
        # self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=self.language, task="transcribe")
        self.bos_token_id = self.hf_model.config.decoder_start_token_id
        self.max_length = self.hf_model.config.max_length

        if not hasattr(self.hf_model, 'max_length'):
            print(f"[INFO] Setting max length to {self.max_length}")
            self.hf_model.max_length = self.max_length

        self.model = pipeline(
            "automatic-speech-recognition",
            model=self.hf_model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=device
        )
        print("[INFO] STT service started...")

    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language: str, **generation_config: dict) -> Tuple[str, str]:
        # Ensure audio is mono channel
        if len(audio_array.shape) > 1:
            print("[INFO] Averaging channels to convert to mono")
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

        lang_code, lang_error = self.get_lang_code(language)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=lang_code, task="transcribe")
        generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

        transcription = self.model(
            {"sampling_rate": 16_000, "raw": audio_array},
            **generation_config,
            generate_kwargs=generate_kwargs
        )

        return transcription["text"].strip(), language
