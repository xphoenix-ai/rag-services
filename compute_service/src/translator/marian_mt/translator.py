import os
import torch
from accelerate import disk_offload
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer, AutoConfig, BitsAndBytesConfig, TextStreamer

from utils.unicode_converter import sinhala_to_singlish
from src.translator.translator_base import TranslatorBase


class Translator(TranslatorBase):
    LANG_MAP = {
        "en": "eng",
        "si": "sin",
        "sing": "eng"
    }

    def __init__(self, en_mul_translator_path: str, mul_en_translator_path: str, translitarator_path: str, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit: bool=False, load_in_8bit: bool=False, device="cuda"):
        self.tr_model_en_mul = None
        self.tr_model_mul_en = None
        
        self.tr_tokenizer_en_mul= None
        self.tr_tokenizer_mul_en = None
        self.tr_tokenizer_singlish = None
        
        self.tr_streamer_en_mul = None
        self.tr_streamer_mul_en = None
        self.tr_streamer_singlish = None
        
        self.device = device if torch.cuda.is_available() else "cpu"
        super().__init__(en_mul_translator_path, mul_en_translator_path, translitarator_path, torch_dtype, low_cpu_mem_usage, load_in_4bit, load_in_8bit)
        
    def _load_model(self, en_mul_translator_path: str, mul_en_translator_path: str, translitarator_path: str, torch_dtype, low_cpu_mem_usage: bool, load_in_4bit: bool, load_in_8bit: bool) -> None:
        self.tr_tokenizer_singlish = AutoTokenizer.from_pretrained(translitarator_path, token=os.getenv("HF_TOKEN"))
        self.tr_tokenizer_en_mul = MarianTokenizer.from_pretrained(en_mul_translator_path, token=os.getenv("HF_TOKEN"))
        self.tr_tokenizer_mul_en = MarianTokenizer.from_pretrained(mul_en_translator_path, token=os.getenv("HF_TOKEN"))
        
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit
            )
        else:
            torch_dtype = torch.float32
            quantization_config = None
            
        self.tr_model_singlish = AutoModelForSeq2SeqLM.from_pretrained(translitarator_path, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, quantization_config=quantization_config, token=os.getenv("HF_TOKEN")).to(self.device)
        # disk_offload(model=self.tr_model_singlish, offload_dir="tr_model_singlish")
        self.tr_model_en_mul = AutoModelForSeq2SeqLM.from_pretrained(en_mul_translator_path, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, quantization_config=quantization_config, token=os.getenv("HF_TOKEN")).to(self.device)
        # disk_offload(model=self.tr_model_en_mul, offload_dir="tr_model_en_mul")
        self.tr_model_mul_en = AutoModelForSeq2SeqLM.from_pretrained(mul_en_translator_path, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, quantization_config=quantization_config, token=os.getenv("HF_TOKEN")).to(self.device)
        # disk_offload(model=self.tr_model_mul_en, offload_dir="tr_model_mul_en")

        self.tr_streamer_singlish = TextStreamer(self.tr_tokenizer_singlish, skip_prompt=True, skip_special_tokens=True)
        self.tr_streamer_en_mul = TextStreamer(self.tr_tokenizer_en_mul, skip_prompt=True, skip_special_tokens=True)
        self.tr_streamer_mul_en = TextStreamer(self.tr_tokenizer_mul_en, skip_prompt=True, skip_special_tokens=True)
        print("[INFO] Translation service started...")
        
    def __translate(self, model, tokenizer, query: str, src_lang_code: str, tgt_lang_code: str, streamer=None, use_min_length: bool=False, is_translitarator: bool=False) -> str:
        if use_min_length:
            query = self.preprocess(query)
        
        if is_translitarator:
            tokenizer.src_lang = src_lang_code    
        else:
            if tgt_lang_code != "eng":
                query = f">>{tgt_lang_code}<< {query}"
            
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        in_len = inputs.input_ids.shape[-1]
        
        # min_length = in_len if tgt_lang_code != "eng_Latn" else None
        # min_length = in_len if use_min_length else None
        min_length = None
        
        if is_translitarator:
            translated_tokens = model.generate(
                **inputs, forced_bos_token_id=tokenizer.encode(tgt_lang_code)[1], min_length=min_length,  max_length=3 * in_len, streamer=streamer, pad_token_id=tokenizer.eos_token_id,
                # do_sample=True, temperature=0.1, top_p=0.95, top_k=20, repetition_penalty=1.0
            )
        else:
            translated_tokens = model.generate(
                **inputs, streamer=streamer,
                # do_sample=True, temperature=0.1, top_p=0.95, top_k=20, repetition_penalty=1.0
            )
        torch.cuda.empty_cache()
        
        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        if use_min_length:
            result = self.postprocess(result)
        
        return result

    def singlish_to_sinhala(self, sing_query: str) -> str:
        """sing --> si"""
        # sing --> si
        si_response = self.__translate(self.tr_model_singlish, self.tr_tokenizer_singlish, sing_query, "eng_Latn", "sin_Sinh", streamer=None, is_translitarator=True)

        return si_response
    
    def english_to_sinhala(self, en_query: str) -> str:
        """en --> si"""
        # en --> si
        si_response = self.__translate(self.tr_model_en_mul, self.tr_tokenizer_en_mul, en_query, "eng", "sin", streamer=None, use_min_length=True)
        
        return si_response
    
    def sinhala_to_english(self, si_query: str) -> str:
        """si --> en"""
        # si --> en
        en_response = self.__translate(self.tr_model_mul_en, self.tr_tokenizer_mul_en, si_query, "sin", "eng", streamer=None, use_min_length=True)
        
        return en_response
    
    def is_ready(self) -> bool:
        if (self.tr_model_singlish is not None) and (self.tr_model_en_mul is not None) and (self.tr_model_mul_en is not None):
            return True
        return False
