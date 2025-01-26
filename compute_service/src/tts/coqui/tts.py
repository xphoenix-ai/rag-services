import numpy as np
from typing import Tuple
from TTS.api import TTS as CoquiTTS

from src.tts.tts_base import TTSBase


class TTS(TTSBase):
    """Coqui TTS implementation supporting multi-speaker and multilingual synthesis."""

    def __init__(self, model_path: str = "tts_models/en/ljspeech/vits", 
                 progress_bar: bool = False, device: str = "cpu", **model_kwargs) -> None:
        """Initialize Coqui TTS model.

        Args:
            model_path (str, optional): Path to the TTS model. Defaults to "tts_models/en/ljspeech/vits".
            progress_bar (bool, optional): Whether to show progress bar during inference. Defaults to False.
            device (str, optional): Device to load model on. Defaults to "cpu".
            **model_kwargs: Additional model configuration parameters
        """
        self.model = None
        super().__init__("coqui", model_path, progress_bar, device, **model_kwargs)
    
    def _load_model(self, model_path: str, progress_bar: bool, device: str, 
                   **model_kwargs) -> None:
        """Load Coqui TTS model and configurations.

        Args:
            model_path (str): Path to the TTS model
            progress_bar (bool): Whether to show progress bar during inference
            device (str): Device to load model on
            **model_kwargs: Additional model configuration parameters
        """
        self.model = CoquiTTS(model_name=model_path, progress_bar=progress_bar, **model_kwargs).to(device)
        self.config = self.model.synthesizer.tts_config
        self.sr = self.config.audio.sample_rate
        self.is_multi_speaker = self.model.is_multi_speaker
        self.is_multilingual = self.model.is_multi_lingual

        print(f"is_multi_speaker: {self.model.is_multi_speaker}")
        print(f"is_multilingual: {self.model.is_multi_lingual}")
        print(f"speakers: {self.model.speakers}")
        print(f"languages: {self.model.languages}")

        if self.model.is_multi_speaker:
            print(f"num_speakers: {len(self.model.speakers)}")

        if self.model.is_multi_lingual:
            print(f"num_languages: {len(self.model.languages)}")
        print("[INFO] TTS service started...")

    def synthesize_one(self, text: str, language: str = None, speaker: str = None, 
                      speaker_wav: str = None) -> Tuple[int, np.ndarray]:
        """Synthesize audio from a single text input.

        Args:
            text (str): Text to synthesize
            language (str, optional): Language code for multilingual models. Defaults to None.
            speaker (str, optional): Speaker ID for multi-speaker models. Defaults to None.
            speaker_wav (str, optional): Path to reference speaker audio for voice cloning. Defaults to None.

        Returns:
            Tuple[int, np.ndarray]: Tuple containing (sample_rate, audio_array)
        """
        audio_arr = self.model.tts(
            text, 
            speaker=speaker if self.is_multi_speaker else None, 
            language=self.get_lang_code(language)[0] if self.is_multilingual else None,
            speaker_wav=speaker_wav
        )
        
        return self.sr, np.array(audio_arr)
