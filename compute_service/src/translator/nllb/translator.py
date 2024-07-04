import os
import torch
from accelerate import disk_offload
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, TextStreamer

from utils.unicode_converter import sinhala_to_singlish
from src.translator.translator_base import TranslatorBase


class Translator(TranslatorBase):
    LANG_MAP = {
        "en": "eng_Latn",
        "si": "sin_Sinh",
        "sing": "eng_Latn"
    }

    def __init__(self, translator_path: str, translitarator_path: str, load_in_4bit: bool=False, load_in_8bit: bool=False):
        self.tr_tokenizer_si = None
        self.tr_tokenizer_en = None
        self.tr_streamer_en = None
        self.tr_streamer_si = None
        super().__init__(translator_path, translitarator_path, load_in_4bit, load_in_8bit)
        
    def _load_model(self, translator_path: str, translitarator_path: str, load_in_4bit: bool, load_in_8bit: bool) -> None:
        self.tr_tokenizer_si = AutoTokenizer.from_pretrained(translator_path, use_auth_token=True, src_lang="sin_Sinh")
        self.tr_tokenizer_en = AutoTokenizer.from_pretrained(translator_path, use_auth_token=True, src_lang="eng_Latn")
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit
            )
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            quantization_config = None
            
        self.tr_model_singlish = AutoModelForSeq2SeqLM.from_pretrained(translitarator_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype, quantization_config=quantization_config, device_map="auto", token=os.getenv("HF_TOKEN"))
        # disk_offload(model=self.tr_model_singlish, offload_dir="tr_model_singlish")
        self.tr_model = AutoModelForSeq2SeqLM.from_pretrained(translator_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype, quantization_config=quantization_config, device_map="auto", token=os.getenv("HF_TOKEN"))
        # disk_offload(model=self.tr_model, offload_dir="tr_model")

        self.tr_streamer_en = TextStreamer(self.tr_tokenizer_en, skip_prompt=True, skip_special_tokens=True)
        self.tr_streamer_si = TextStreamer(self.tr_tokenizer_si, skip_prompt=True, skip_special_tokens=True)
        print("[INFO] Translation service started...")
        
    def __translate(self, model, tokenizer, query: str, tgt_lang_code: str, streamer=None, use_min_length: bool=False) -> str:
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        in_len = inputs.input_ids.shape[-1]
        # min_length = in_len if tgt_lang_code != "eng_Latn" else None
        min_length = in_len if use_min_length else None
        
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.encode(tgt_lang_code)[1], min_length=min_length,  max_length=3 * in_len, streamer=streamer, pad_token_id=tokenizer.eos_token_id,
            do_sample=True, temperature=0.1, top_p=0.95, top_k=20, repetition_penalty=1.0
        )
        torch.cuda.empty_cache()
        
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    def singlish_to_sinhala(self, sing_query: str) -> str:
        """sing --> si"""
        # sing --> si
        si_response = self.__translate(self.tr_model_singlish, self.tr_tokenizer_en, sing_query, "sin_Sinh", streamer=None)

        return si_response
    
    def english_to_sinhala(self, en_query: str) -> str:
        """en --> si"""
        # en --> si
        si_response = self.__translate(self.tr_model, self.tr_tokenizer_en, en_query, "sin_Sinh", streamer=None, use_min_length=True)
        
        return si_response
    
    def sinhala_to_english(self, si_query: str) -> str:
        """si --> en"""
        # si --> en
        en_response = self.__translate(self.tr_model, self.tr_tokenizer_si, si_query, "eng_Latn", streamer=None)
        
        return en_response
