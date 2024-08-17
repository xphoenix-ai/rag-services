import os
import requests
import numpy as np
from typing import Tuple


class TTS:
    def __init__(self):
        self.url = os.getenv("TTS_URL")
        
    def synthesize(self, text: str, language: str=None) -> Tuple[int, list]:
        json_body = {
            "text": text,
            "language": language
        }
        response = requests.post(self.url, json=json_body)
        response = response.json()

        audio_array = response["audio_response"]
        sample_rate = response["sample_rate"]
        
        return sample_rate, audio_array
    
    def is_ready(self) -> bool:
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["tts"]
        except:
            status = False
            
        return status
