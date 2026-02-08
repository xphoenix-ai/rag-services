import os
import requests


class Translator:
    """Translation service that communicates with a translator server via HTTP."""
    
    def __init__(self):
        """Initialize the Translator service.
        
        Reads the translator service URL from the TRANSLATOR_URL environment variable.
        """
        self.url = os.getenv("TRANSLATOR_URL")
        
    def translate(self, query, src_lang, tgt_lang, trace_lf):
        """Translate text from source language to target language.
        
        Args:
            query (str): Text to translate
            src_lang (str): Source language code
            tgt_lang (str): Target language code
            trace_lf: Optional Langfuse tracer for logging the translation operation
            
        Returns:
            str: Translated text in the target language
            
        Raises:
            requests.RequestException: If the HTTP request to the server fails
        """
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
        """Check if the translator service is available and ready.
        
        Returns:
            bool: True if the translator service is ready, False otherwise
        """
        try:
            response = requests.get(os.getenv("STATUS_URL"))
            status = response.json()["translator"]
        except:
            status = False
            
        return status
