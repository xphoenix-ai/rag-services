import os
import requests
import numpy as np
from typing import Tuple


class TTS:
    """Text-to-Speech service that communicates with a TTS server via HTTP."""
    
    def __init__(self):
        """Initialize the TTS service.
        
        Reads the TTS service URL from the TTS_URL environment variable.
        """
        self.url = os.getenv("TTS_URL")
        
    def synthesize(self, text: str, language: str=None) -> Tuple[int, list]:
        """Synthesize text to audio.
        
        Args:
            text (str): Text to synthesize into audio
            language (str, optional): Language code for synthesis. Defaults to None.
            
        Returns:
            Tuple[int, list]: Tuple containing (sample_rate, audio_array) where
                sample_rate is an integer and audio_array is a list of audio samples
                
        Raises:
            requests.RequestException: If the HTTP request to the server fails
        """
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
        """Check if the TTS service is available and ready.
        
        Returns:
            bool: True if the TTS service is ready, False otherwise
        """
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["tts"]
        except:
            status = False
            
        return status
