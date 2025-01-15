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

    def __init__(self, en_mul_translator_path: str, mul_en_translator_path: str, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_4bit: bool=False, load_in_8bit: bool=False, device="cuda"):
        self.tr_model_en_mul = None
        self.tr_model_mul_en = None
        
        self.tr_tokenizer_en_mul= None
        self.tr_tokenizer_mul_en = None

        self.tr_streamer_en_mul = None
        self.tr_streamer_mul_en = None

        self.device = device if torch.cuda.is_available() else "cpu"
        super().__init__("marian_mt", en_mul_translator_path, mul_en_translator_path, torch_dtype, low_cpu_mem_usage, load_in_4bit, load_in_8bit)
        
    def _load_model(self, en_mul_translator_path: str, mul_en_translator_path: str, translitarator_path: str, torch_dtype, low_cpu_mem_usage: bool, load_in_4bit: bool, load_in_8bit: bool) -> None:
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
            
        self.tr_model_en_mul = AutoModelForSeq2SeqLM.from_pretrained(en_mul_translator_path, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, quantization_config=quantization_config, token=os.getenv("HF_TOKEN")).to(self.device)
        self.tr_model_mul_en = AutoModelForSeq2SeqLM.from_pretrained(mul_en_translator_path, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, quantization_config=quantization_config, token=os.getenv("HF_TOKEN")).to(self.device)

        self.tr_streamer_en_mul = TextStreamer(self.tr_tokenizer_en_mul, skip_prompt=True, skip_special_tokens=True)
        self.tr_streamer_mul_en = TextStreamer(self.tr_tokenizer_mul_en, skip_prompt=True, skip_special_tokens=True)
        print("[INFO] Translation service started...")

    def __half_translate(self, model, tokenizer, query: str, src_lang_code: str, tgt_lang_code: str, streamer=None, use_min_length: bool=False) -> str:

        if use_min_length:
            query = self.preprocess(query)

        if tgt_lang_code != "eng":
            query = f">>{tgt_lang_code}<< {query}"
            
        inputs = tokenizer(query, return_tensors="pt").to(model.device)

        translated_tokens = model.generate(
            **inputs, streamer=streamer,
        )
        torch.cuda.empty_cache()
        
        result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        if use_min_length:
            result = self.postprocess(result)
        
        return result

    def translate(self, query: str, src_lang: str, tgt_lang: str, use_min_length: bool=False):
        src_lang_code, src_lang_error = self.get_lang_code(src_lang)
        tgt_lang_code, tgt_lang_error = self.get_lang_code(tgt_lang)
        error = src_lang_error + tgt_lang_error

        if error:
            return "", error

        if src_lang_code == tgt_lang_code:
            return query, ""
        elif src_lang_code ==  "eng":
            res = self.__half_translate(self.tr_model_en_mul, self.tr_tokenizer_en_mul, query, "eng", tgt_lang_code, streamer=self.tr_streamer_en_mul, use_min_length=True)
            return res, ""
        else:  # src_lang_code != "eng"
            inter_query = self.__half_translate(self.tr_model_mul_en, self.tr_tokenizer_mul_en, query, src_lang_code, "eng", streamer=self.tr_streamer_mul_en, use_min_length=True)

            if tgt_lang_code == "eng":
                return inter_query, ""
            else:
                res = self.__half_translate(self.tr_model_en_mul, self.tr_tokenizer_en_mul, inter_query, "eng", tgt_lang_code, streamer=self.tr_streamer_en_mul, use_min_length=True)
                return res, ""

    def is_ready(self) -> bool:
        if (self.tr_model_en_mul is not None) and (self.tr_model_mul_en is not None):
            return True
        return False
