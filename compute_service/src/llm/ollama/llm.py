import os
import torch
import ollama
from ollama import generate, chat, show, pull


from src.llm.llm_base import LLMBase


class LLM(LLMBase):
    def __init__(
        self, 
        model_path="llama3:instruct",
        keep_alive="5m",
        tokenize_with_chat_template=True
        ):
        self.model = None
        self.model_path = model_path
        self.keep_alive = keep_alive
        self.tokenize_with_chat_template = tokenize_with_chat_template
        self.default_generation_config = {
            "num_predict": 200,
            "top_k": 20,
            "top_p": 0.95,
            "temperature": 0.1,            
        }
        super().__init__("ollama", model_path)
        
    def _load_model(self, model_path):
        try:
            ollama.show(model_path)
        except ollama.ResponseError as e:
            print(f"[ERROR] {e}")
            
            if e.status_code == 404:
                print(f"[INFO] Downloading the model: {model_path}...")
                ollama.pull(model_path)
        self.model = 1
        print("[INFO] LLM service started...")
                
    def generate(self, prompt, **generation_config):
        torch.cuda.empty_cache()
        
        full_generation_config = self.default_generation_config
        full_generation_config.update(generation_config)
        
        if self.tokenize_with_chat_template:
            response = chat(
                model=self.model_path, 
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
                keep_alive=self.keep_alive,
                options=full_generation_config
            )
        else:
            response = generate(
                model=self.model_path,
                prompt=prompt,
                keep_alive=self.keep_alive,
                options=full_generation_config 
            )
        result = response['message']['content']
        torch.cuda.empty_cache()
        
        return result
