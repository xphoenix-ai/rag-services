import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.llm.llm_base import LLMBase


class LLM(LLMBase):
    """Hugging Face Transformers-based LLM implementation."""

    def __init__(self, model_path: str, torch_dtype: torch.dtype = torch.float16, 
                 low_cpu_mem_usage: bool = True, load_in_4bit: bool = False, 
                 load_in_8bit: bool = False, tokenize_with_chat_template: bool = True) -> None:
        """Initialize Hugging Face LLM.

        Args:
            model_path (str): Path to the model
            torch_dtype (torch.dtype, optional): Data type for model weights. Defaults to torch.float16.
            low_cpu_mem_usage (bool, optional): Whether to use low CPU memory usage. Defaults to True.
            load_in_4bit (bool, optional): Whether to load model in 4-bit precision. Defaults to False.
            load_in_8bit (bool, optional): Whether to load model in 8-bit precision. Defaults to False.
            tokenize_with_chat_template (bool, optional): Whether to use chat template for tokenization. Defaults to True.
        """
        self.model = None
        self.tokenizer = None
        self.device = None
        self.tokenize_with_chat_template = tokenize_with_chat_template
        # self.init(model_path, torch_dtype, low_cpu_mem_usage, load_in_4bit, load_in_8bit)
        super().__init__("hf", model_path, torch_dtype, low_cpu_mem_usage, load_in_4bit, load_in_8bit)
        
    def _load_model(self, model_path: str, torch_dtype: torch.dtype, 
                   low_cpu_mem_usage: bool, load_in_4bit: bool, load_in_8bit: bool) -> None:
        """Load Hugging Face model and tokenizer.

        Args:
            model_path (str): Path to the model
            torch_dtype (torch.dtype): Data type for model weights
            low_cpu_mem_usage (bool): Whether to use low CPU memory usage
            load_in_4bit (bool): Whether to load model in 4-bit precision
            load_in_8bit (bool): Whether to load model in 8-bit precision
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", token=os.getenv("HF_TOKEN"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit
            )
        else:
            torch_dtype = torch.float32
            quantization_config = None
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            token=os.getenv("HF_TOKEN")
        )
        self.device = self.model.device
        print("[INFO] LLM service started...")
                
    def generate(self, prompt: str, **generation_config: dict) -> str:
        """Generate text using the LLM.

        Args:
            prompt (str): Input prompt for generation
            **generation_config: Generation configuration parameters

        Returns:
            str: Generated text response
        """
        torch.cuda.empty_cache()
        if self.tokenize_with_chat_template:
            messages = [
                {"role": "user", "content": prompt},
            ]
            inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(self.device)
            inputs = {'input_ids': inputs}
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            
        gen_tokens = self.model.generate(
            **inputs,
            **generation_config,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        prompt_tokens = len(inputs['input_ids'][0])
        gen_tokens = gen_tokens[0, prompt_tokens:]
        result = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        torch.cuda.empty_cache()
        
        return result
