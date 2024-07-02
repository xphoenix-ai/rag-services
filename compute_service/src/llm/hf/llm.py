import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.llm.llm_base import LLMBase


class LLM(LLMBase):
    def __init__(self, model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=False, load_in_8bit=False, tokenize_with_chat_template=True):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.tokenize_with_chat_template = tokenize_with_chat_template
        # self.init(model_path, torch_dtype, low_cpu_mem_usage, load_in_4bit, load_in_8bit)
        super().__init__(model_path, torch_dtype, low_cpu_mem_usage, load_in_4bit, load_in_8bit)
        
    def _load_model(self, model_path, torch_dtype, low_cpu_mem_usage, load_in_4bit, load_in_8bit):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", token=os.getenv("HF_TOKEN"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit
            )
        else:
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
                
    def generate(self, prompt, **generation_config):
        torch.cuda.empty_cache()
        if self.tokenize_with_chat_template:
            messages = [
                {"role": "user", "content": prompt},
            ]
            inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(self.device)
            inputs = {'input_ids': inputs}  # For compatibility with generate()
        else:
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            
        gen_tokens = self.model.generate(
            **inputs,
            **generation_config,
            pad_token_id=self.tokenizer.eos_token_id,   # To prevent warning
        )

        prompt_tokens = len(inputs['input_ids'][0])
        gen_tokens = gen_tokens[0, prompt_tokens:]  # Remove the prompt tokens
        result = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        torch.cuda.empty_cache()
        
        return result
