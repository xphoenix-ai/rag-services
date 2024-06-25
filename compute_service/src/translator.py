import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, TextStreamer

from utils.unicode_converter import sinhala_to_singlish


class Translator:
    LANG_MAP = {
        "en": "eng_Latn",
        "si": "sin_Sinh",
        "sing": "eng_Latn"
    }

    def __init__(self, translator_path, translitarator_path, load_in_4bit=False, load_in_8bit=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tr_tokenizer_si = AutoTokenizer.from_pretrained(translator_path, use_auth_token=True, src_lang="sin_Sinh")
        self.tr_tokenizer_en = AutoTokenizer.from_pretrained(translator_path, use_auth_token=True, src_lang="eng_Latn")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )
        self.tr_model_singlish = AutoModelForSeq2SeqLM.from_pretrained(translitarator_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, quantization_config=quantization_config, device_map="auto", token=os.getenv("HF_TOKEN"))
        self.tr_model = AutoModelForSeq2SeqLM.from_pretrained(translator_path, use_auth_token=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, quantization_config=quantization_config, device_map="auto")

        self.tr_streamer_en = TextStreamer(self.tr_tokenizer_en, skip_prompt=True, skip_special_tokens=True)
        self.tr_streamer_si = TextStreamer(self.tr_tokenizer_si, skip_prompt=True, skip_special_tokens=True)
        print("[INFO] Translation service started...")
        
    def __translate(self, model, tokenizer, query, tgt_lang_code, streamer=None):
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        in_len = inputs.input_ids.shape[-1]
        min_length = in_len if tgt_lang_code != "eng_Latn" else None
        
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code], min_length=min_length,  max_length=3 * in_len, streamer=streamer, pad_token_id=tokenizer.eos_token_id,
            do_sample=True, temperature=0.1, top_p=0.95, top_k=20, repetition_penalty=1.0
        )
        torch.cuda.empty_cache()
        
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    def singlish_to_sinhala(self, sing_query):
        """sing --> si"""
        # sing --> si
        si_response = self.__translate(self.tr_model_singlish, self.tr_tokenizer_en, sing_query, "sin_Sinh", streamer=None)

        return si_response
    
    def english_to_sinhala(self, en_query):
        """en --> si"""
        # en --> si
        si_response = self.__translate(self.tr_model, self.tr_tokenizer_en, en_query, "sin_Sinh", streamer=None)
        
        return si_response
    
    def sinhala_to_english(self, si_query):
        """si --> en"""
        # si --> en
        en_response = self.__translate(self.tr_model, self.tr_tokenizer_si, si_query, "eng_Latn", streamer=None)
        
        return en_response
    
    def sinhala_to_singlish(self, si_query):
        """si --> sing"""
        # si --> sing
        sing_response = sinhala_to_singlish(si_query)
        
        return sing_response
    
    def singlish_to_english(self, sing_query):
        """sing --> en"""
        # sing --> si
        si_response = self.singlish_to_sinhala(sing_query)
        
        # si --> en
        en_response = self.sinhala_to_english(si_response)
        
        return en_response
    
    def english_to_singlish(self, en_query):
        """sing --> en"""
        # en --> si
        si_response = self.english_to_sinhala(en_query)
        
        # si --> sing
        sing_response = self.sinhala_to_singlish(si_response)
        
        return sing_response
        