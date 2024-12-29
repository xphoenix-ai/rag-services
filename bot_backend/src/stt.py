import os
import requests
from typing import Tuple


class STT:
    def __init__(self):
        self.url = os.getenv("STT_URL")
        
    def transcribe(self, audio_data: list, sample_rate: int, src_lang: str=None, tgt_lang: str=None) -> Tuple[str, str]:
        json_body = {
            "audio_data": audio_data,
            "sample_rate": sample_rate
        }
        response = requests.post(self.url, json=json_body)
        response = response.json()

        return response["transcription"], response["language"]
    
    def is_ready(self) -> bool:
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["stt"]
        except:
            status = False
            
        return status
