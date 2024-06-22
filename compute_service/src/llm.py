import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LLM:
    def __init__(self, model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit=False, load_in_8bit=False, tokenize_with_chat_template=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", token=os.getenv("HF_TOKEN"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            token=os.getenv("HF_TOKEN")
        )
        self.device = self.model.device
        self.tokenize_with_chat_template = tokenize_with_chat_template
        print("[INFO] LLM service started...")
                
    def generate(self, prompt, do_sample=True, max_new_tokens=200, top_k=20, top_p=0.95, temperature=0.1, repetition_penalty=1.0):
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
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,   # To prevent warning
            repetition_penalty=repetition_penalty,
        )

        prompt_tokens = len(inputs['input_ids'][0])
        gen_tokens = gen_tokens[0, prompt_tokens:]  # Remove the prompt tokens
        result = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        torch.cuda.empty_cache()
        
        return result
