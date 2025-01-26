import os
import torch
import ollama
from ollama import generate, chat, show, pull


from src.llm.llm_base import LLMBase


class LLM(LLMBase):
    """Ollama-based LLM implementation."""

    def __init__(self, model_path: str = "llama3:instruct", keep_alive: str = "5m",
                 tokenize_with_chat_template: bool = True) -> None:
        """Initialize Ollama LLM.

        Args:
            model_path (str, optional): Path/name of the Ollama model. Defaults to "llama3:instruct".
            keep_alive (str, optional): Duration to keep model loaded. Defaults to "5m".
            tokenize_with_chat_template (bool, optional): Whether to use chat template. Defaults to True.
        """
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
        
    def _load_model(self, model_path: str) -> None:
        """Load Ollama model.

        Args:
            model_path (str): Path/name of the Ollama model
        """
        try:
            ollama.show(model_path)
        except ollama.ResponseError as e:
            print(f"[ERROR] {e}")
            
            if e.status_code == 404:
                print(f"[INFO] Downloading the model: {model_path}...")
                ollama.pull(model_path)
        self.model = 1
        print("[INFO] LLM service started...")
                
    def generate(self, prompt: str, **generation_config: dict) -> str:
        """Generate text using Ollama model.

        Args:
            prompt (str): Input prompt for generation
            **generation_config: Generation configuration parameters

        Returns:
            str: Generated text response
        """
        torch.cuda.empty_cache()
        
        full_generation_config = self.default_generation_config.copy()
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
