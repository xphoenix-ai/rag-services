import os
import requests


class Translator:
    def __init__(self):
        self.url = os.getenv("TRANSLATOR_URL")
        
    def translate(self, query, src_lang, tgt_lang, trace_lf):
        json_body = {
            "src": query,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        response = requests.post(self.url, json=json_body)
        response = response.json()

        if trace_lf is not None:
            span = trace_lf.span(
                name = f"translator_{src_lang}-{tgt_lang}",
                input = query,
                output = {"target":response["tgt"],
                          "intermediate_res":response["intermediate_res"]}
                ) 

        return response["tgt"]
    
    def is_ready(self):
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["translator"]
        except:
            status = False
            
        return status
