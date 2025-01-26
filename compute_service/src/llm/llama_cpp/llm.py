import os
import torch
from llama_cpp import Llama

from src.llm.llm_base import LLMBase


class LLM(LLMBase):
    """LlamaCpp-based LLM implementation supporting local inference with GGUF models."""

    def __init__(self, model_path: str = "bartowski/Meta-Llama-3-8B-Instruct-GGUF", 
                 filename: str = "*Q4_K_M.gguf", chat_format: str = "llama-3",
                 tokenize_with_chat_template: bool = True) -> None:
        """Initialize LlamaCpp LLM.

        Args:
            model_path (str, optional): Path or HF repo ID for the GGUF model. 
                Defaults to "bartowski/Meta-Llama-3-8B-Instruct-GGUF".
            filename (str, optional): Specific GGUF file to load if using HF repo. 
                Defaults to "*Q4_K_M.gguf".
            chat_format (str, optional): Chat template format to use. 
                Defaults to "llama-3".
            tokenize_with_chat_template (bool, optional): Whether to use chat template. 
                Defaults to True.
        """
        self.model = None
        self.tokenize_with_chat_template = tokenize_with_chat_template
        self.default_generation_config = {
            "max_tokens": 200,
            "top_k": 20,
            "top_p": 0.95,
            "temperature": 0.1,         
        }
        super().__init__("llama_cpp", model_path, filename, chat_format)
        
    def __postprocess(self, result: str) -> str:
        """Post-process model output to clean up response format.

        Args:
            result (str): Raw model output

        Returns:
            str: Cleaned model output with system prompts and role labels removed
        """
        print(f"befor postprocess: {result}")
        
        result = [x.strip() for x in result.split('System:') if x.strip()][0]
        result = [x.strip() for x in result.split('Context:') if x.strip()][0]
        result = [x.strip() for x in result.split('User:') if x.strip()][0]
        result = result.replace('Assistant:', '').strip()
        
        return result
        # return result.split('System:')[0].strip().split('Assistant:')[-1].strip()
        # return result.split('System:')[0].strip().replace('Assistant:', '').strip()
        
    def _load_model(self, model_path: str, filename: str, chat_format: str) -> None:
        """Load LlamaCpp model either from local file or Hugging Face repo.

        Args:
            model_path (str): Path or HF repo ID for the GGUF model
            filename (str): Specific GGUF file to load if using HF repo
            chat_format (str): Chat template format to use
        """
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
                
    def generate(self, prompt: str, **generation_config: dict) -> str:
        """Generate text using LlamaCpp model.

        Args:
            prompt (str): Input prompt for generation
            **generation_config: Additional generation parameters that override defaults.
                Supported parameters include:
                - max_tokens (int): Maximum number of tokens to generate
                - top_k (int): Number of highest probability tokens to consider
                - top_p (float): Cumulative probability threshold for token sampling
                - temperature (float): Sampling temperature for controlling randomness

        Returns:
            str: Generated text response with system prompts and role labels removed
        """
        torch.cuda.empty_cache()
        
        full_generation_config = self.default_generation_config.copy()
        full_generation_config.update(generation_config)
        
        if self.tokenize_with_chat_template:
            messages = [
                    {"role": "user", "content": prompt},
                ]
            output = self.model.create_chat_completion(
                messages=messages,
                **full_generation_config
            )
        else:
            output = self.model(
                prompt,
                echo=False,
                **full_generation_config
            )
            
        result = output["choices"][-1]["message"]["content"]
        torch.cuda.empty_cache()
        
        # return result
        return self.__postprocess(result)
