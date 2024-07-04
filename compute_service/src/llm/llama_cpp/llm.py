import os
import torch
from llama_cpp import Llama

from src.llm.llm_base import LLMBase


class LLM(LLMBase):
    def __init__(
        self, 
        model_path="bartowski/Meta-Llama-3-8B-Instruct-GGUF", 
        filename="*Q4_K_M.gguf", 
        chat_format="llama-3",
        tokenize_with_chat_template=True
        ):
        self.model = None
        self.tokenize_with_chat_template = tokenize_with_chat_template
        self.default_generation_config = {
            "max_tokens": 200,
            "top_k": 20,
            "top_p": 0.95,
            "temperature": 0.1,            
        }
        super().__init__(model_path, filename, chat_format)
        
    def _load_model(self, model_path, filename, chat_format):
        if os.path.isfile(model_path):
            self.model = Llama(
                model_path=model_path, 
                chat_format=chat_format,
                n_ctx=4096,
            )
        else:
            self.model = Llama.from_pretrained(
                repo_id=model_path,
                filename=filename,
                verbose=False,
                chat_format=chat_format,
                n_ctx=4096,
            )
        print("[INFO] LLM service started...")
                
    def generate(self, prompt, **generation_config):
        torch.cuda.empty_cache()
        
        # generation_config.update(self.default_generation_config)
        generation_config = self.default_generation_config
        
        if self.tokenize_with_chat_template:
            messages = [
                    {"role": "user", "content": prompt},
                ]
            output = self.model.create_chat_completion(
                messages=messages,
                **generation_config
            )
        else:
            output = self.model(
                prompt,
                **generation_config
            )
            
        result = output["choices"][-1]["message"]["content"]
        torch.cuda.empty_cache()
        
        return result
