import os
import requests
from typing import Tuple


class STT:
    """Speech-to-Text service that communicates with an STT server via HTTP."""
    
    def __init__(self):
        """Initialize the STT service.
        
        Reads the STT service URL from the STT_URL environment variable.
        """
        self.url = os.getenv("STT_URL")
        
    def transcribe(self, audio_data: list, sample_rate: int, src_lang: str=None, tgt_lang: str=None) -> Tuple[str, str]:
        """Transcribe audio to text.
        
        Args:
            audio_data (list): Audio data as a list of samples
            sample_rate (int): Sample rate of the audio in Hz
            src_lang (str, optional): Source language code. Defaults to None.
            tgt_lang (str, optional): Target language code. Defaults to None.
            
        Returns:
            Tuple[str, str]: Tuple containing (transcribed_text, detected_language)
            
        Raises:
            requests.RequestException: If the HTTP request to the server fails
        """
        json_body = {
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "language": src_lang
        }
        response = requests.post(self.url, json=json_body)
        response = response.json()

        return response["transcription"], response["language"]
    
    def is_ready(self) -> bool:
        """Check if the STT service is available and ready.
        
        Returns:
            bool: True if the STT service is ready, False otherwise
        """
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["stt"]
        except:
            status = False
            
        return status
