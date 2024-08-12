import numpy as np
from typing import Tuple
from TTS.api import TTS as CoquiTTS

from src.tts.tts_base import TTSBase


class TTS(TTSBase):
    def __init__(self, model_path: str="tts_models/en/ljspeech/vits", progress_bar: bool=False, device: str="cpu", **model_kwargs):
        self.model = None
        super().__init__(model_path, progress_bar, device, **model_kwargs)
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
    
    def _load_model(self, model_path, progress_bar: bool, device: str, **model_kwargs) -> None:
        self.model = CoquiTTS(model_name=model_path, progress_bar=progress_bar, **model_kwargs).to(device)

    def synthesize_one(self, text: str, language: str=None, speaker: str=None, speaker_wav: str=None) -> Tuple[int, np.ndarray]:
        audio_arr = self.model.tts(
            text, 
            speaker=speaker if self.is_multi_speaker else None, 
            language=language if self.is_multilingual else None, 
            speaker_wav=speaker_wav
        )
        
        return self.sr, np.array(audio_arr)
