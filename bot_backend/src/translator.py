import os
import requests


class Translator:
    def __init__(self):
        self.url = os.getenv("TRANSLATOR_URL")
        
    def translate(self, query, src_lang, tgt_lang):
        json_body = {
            "src": query,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        response = requests.post(self.url, json=json_body)
        response = response.json()

        return response["tgt"]
    
    def is_ready(self):
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["translator"]
        except:
            status = False
            
        return status
