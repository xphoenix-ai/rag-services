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

    def __init__(self, translator_path: str, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit: bool=False, load_in_8bit: bool=False):
        self.tr_tokenizer = None
        self.tr_streamer = None
        super().__init__("nllb", translator_path, torch_dtype, low_cpu_mem_usage, load_in_4bit, load_in_8bit)
        
    def _load_model(self, translator_path: str, torch_dtype, low_cpu_mem_usage: bool, load_in_4bit: bool, load_in_8bit: bool) -> None:
        self.tr_tokenizer = AutoTokenizer.from_pretrained(translator_path, token=os.getenv("HF_TOKEN")) #, src_lang="sin_Sinh")
        
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit
            )
        else:
            torch_dtype = torch.float32
            quantization_config = None
            
        self.tr_model = AutoModelForSeq2SeqLM.from_pretrained(translator_path, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, quantization_config=quantization_config, device_map="auto", token=os.getenv("HF_TOKEN"))

        self.tr_streamer = TextStreamer(self.tr_tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("[INFO] Translation service started...")
        
    def translate(self, query: str, src_lang: str, tgt_lang: str, use_min_length: bool=False):
        src_lang_code, src_lang_error = self.get_lang_code(src_lang)
        tgt_lang_code, tgt_lang_error = self.get_lang_code(tgt_lang)
        error = src_lang_error + tgt_lang_error

        if error:
            return "", error

        if src_lang_code == tgt_lang_code:
            return query, ""

        if use_min_length:
            query = self.preprocess(query)
        
        self.tr_tokenizer.src_lang = src_lang_code
        inputs = self.tr_tokenizer(query, return_tensors="pt").to(self.tr_model.device)
        in_len = inputs.input_ids.shape[-1]
        
        min_length = None
        
        translated_tokens = self.tr_model.generate(
            **inputs, forced_bos_token_id=self.tr_tokenizer.encode(tgt_lang_code)[1], min_length=min_length,  max_length=3 * in_len, streamer=self.tr_streamer, pad_token_id=self.tr_tokenizer.eos_token_id,
        )
        torch.cuda.empty_cache()
        
        result = self.tr_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        if use_min_length:
            result = self.postprocess(result)
        
        return result, ""
