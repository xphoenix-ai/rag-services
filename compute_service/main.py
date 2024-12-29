import os
import sys
import pytz
import time
import ngrok
import uvicorn
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, Body
from typing import Union, Optional
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, dotenv_values
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from utils.config_reader import get_config
from utils.get_module_by_name import get_module

from src.llm.llm_base import LLMBase
from src.stt.stt_base import STTBase
from src.tts.tts_base import TTSBase
from src.encoder.encoder_base import EncoderBase
from src.translator.translator_base import TranslatorBase


load_dotenv()

llm_class = os.getenv('LLM_CLASS', 'hf')
encoder_class = os.getenv('ENCODER_CLASS', 'st')
translator_class = os.getenv('TRANSLATOR_CLASS', 'nllb')
stt_class = os.getenv('STT_CLASS', 'whisper')
tts_class = os.getenv('TTS_CLASS', 'coqui')

llm_config = get_config("llm_config.yml")[llm_class]
print(f"LLM Config: {llm_config}")

translator_config = get_config("translator_config.yml")[translator_class]
print(f"Translator Config: {translator_config}")

encoder_config = get_config("encoder_config.yml")[encoder_class]
print(f"Encoder Config: {encoder_config}")

stt_config = get_config("stt_config.yml")[stt_class]
print(f"STT Config: {stt_config}")

tts_config = get_config("tts_config.yml")[tts_class]
print(f"TTS Config: {tts_config}")

LLM = get_module(f"src.llm.{llm_class}.llm", "LLM")
Translator = get_module(f"src.translator.{translator_class}.translator", "Translator")
Encoder = get_module(f"src.encoder.{encoder_class}.encoder", "Encoder")
STT = get_module(f"src.stt.{stt_class}.stt", "STT")
TTS = get_module(f"src.tts.{tts_class}.tts", "TTS")

app = FastAPI()

llm: LLMBase = LLM(**llm_config["model_config"])
stt: STTBase  = STT(**stt_config["model_config"])
tts: TTSBase = TTS(**tts_config["model_config"])
encoder: EncoderBase = Encoder(**encoder_config["model_config"])
translator: TranslatorBase = Translator(**translator_config["model_config"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class TrIntput(BaseModel):
    src: str
    src_lang: str
    tgt_lang: str
    


class TrOutput(BaseModel):
    src: str
    tgt: Optional[str]
    intermediate_res: Optional[str]
    src_lang: str
    tgt_lang: str
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]
        
    
class EmbdIntput(BaseModel):
    sentences: list[str]
    
    
class EmbdOutput(BaseModel):
    sentences: list[str]
    embeddings: Optional[list[list[float]]]
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]
    

# class LLMInput(BaseModel):
#     prompt: str
#     do_sample: bool = True
#     max_new_tokens: int = 200
#     top_k: Optional[int] = 20
#     top_p: Optional[float] = 0.95
#     temperature: Optional[float] = 0.1


class LLMOutput(BaseModel):
    prompt: str
    response: str
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]


class STTOutput(BaseModel):
    transcription: str
    language: str
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]


class TTSOutput(BaseModel):
    audio_response: list
    sample_rate: int
    input_text: str
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]


@app.get("/")
async def read_root() -> JSONResponse:
    return JSONResponse({"info": "Compute Service", "version": os.getenv("TR_VERSION"), "vendor": "XXX"})

@app.post("/translate")
async def translate(tr_item: TrIntput) -> JSONResponse:
    t_start = time.time()
    
    src_lang = tr_item.src_lang
    tgt_lang = tr_item.tgt_lang
    src = tr_item.src
    error = ""
    
    if src_lang == "sing" and tgt_lang == "en":
        res, int_res = translator.singlish_to_english(src)
        # res = await translator.singlish_to_english(src)
    elif src_lang == "sing" and tgt_lang == "si":
        res, int_res = translator.singlish_to_sinhala(src)
        # res = await translator.singlish_to_sinhala(src)
    elif src_lang == "en" and tgt_lang == "si":
        res, int_res = translator.english_to_sinhala(src)
        # res = await translator.english_to_sinhala(src)
    elif src_lang == "en" and tgt_lang == "sing":
        res, int_res = translator.english_to_singlish(src)
        # res = await translator.english_to_singlish(src)
    elif src_lang == "si" and tgt_lang == "sing":
        res, int_res = translator.sinhala_to_singlish(src)
        # res = await translator.sinhala_to_singlish(src)
    elif src_lang == "si" and tgt_lang == "en":
        res, int_res = translator.sinhala_to_english(src)
        # res = await translator.sinhala_to_english(src)
    elif src_lang == tgt_lang:
        res = src
        int_res = ""
    else:
        res = ""
        int_res = ""
        error = f"Not supported language pair: {src_lang} and {tgt_lang}"
        # raise ValueError(f"Not supported language pair: {src_lang} and {tgt_lang}")
    
    t_end = time.time()
    
    res_obj = TrOutput(
    src = src,
    tgt = res,
    intermediate_res = int_res,
    src_lang = src_lang,
    tgt_lang = tgt_lang,
    modified_time = datetime.now(pytz.UTC),
    time_taken = (t_end - t_start),
    error = error
    )
    
    json_obj = jsonable_encoder(res_obj)
    
    return JSONResponse(json_obj)

@app.post("/encode")
async def encode(embed_item: EmbdIntput) -> JSONResponse:
    t_start = time.time()
    
    sentences = embed_item.sentences
    embeddings = encoder.encode(sentences)
    # embeddings = await encoder.encode(sentences)
    error = ""
    
    t_end = time.time()
    res_obj = EmbdOutput(
        sentences = sentences,
        embeddings = embeddings,
        modified_time = datetime.now(pytz.UTC),
        time_taken = (t_end - t_start),
        error = error
    )
    json_obj = jsonable_encoder(res_obj)
    
    return JSONResponse(json_obj)

@app.post("/generate")
# async def generate(llm_input: LLMInput) -> JSONResponse:
async def generate(
    prompt: str=Body(embed=True), 
    generation_config: dict=Body(embed=True, default=llm_config["generation_config"])
    ) -> JSONResponse:
    t_start = time.time()
    
    response = llm.generate(prompt, **generation_config)
    # response = await llm.generate(prompt, **generation_config)
    error = ""
    
    t_end = time.time()
    res_obj = LLMOutput(
        prompt = prompt,
        response = response,
        modified_time = datetime.now(pytz.UTC),
        time_taken = (t_end - t_start),
        error = error
    )
    json_obj = jsonable_encoder(res_obj)
    
    return JSONResponse(json_obj)

@app.post("/transcribe")
async def transcribe(audio_data: list=Body(embed=True), sample_rate: int=Body(embed=True, default=16_000)) -> JSONResponse:
    t_start = time.time()

    audio_array = np.array(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    generation_config = stt_config.get("generation_config", {})
    transcription, language = stt.transcribe(audio_array, sample_rate, **generation_config)
    error = ""
    
    t_end = time.time()
    res_obj = STTOutput(
        transcription = transcription,
        language = language,
        modified_time = datetime.now(pytz.UTC),
        time_taken = (t_end - t_start),
        error = error
    )
    json_obj = jsonable_encoder(res_obj)
    
    return JSONResponse(json_obj)

@app.post("/synthesize")
async def synthesize(text: str=Body(embed=True), language: str=Body(embed=True, default="en")) -> JSONResponse:
    t_start = time.time()

    generation_config = tts_config.get("generation_config", {})
    sample_rate, audio_response = tts.synthesize(text, **generation_config)
    error = ""
    
    t_end = time.time()
    res_obj = TTSOutput(
        audio_response = audio_response.tolist(),
        sample_rate = sample_rate,
        input_text = text,
        modified_time = datetime.now(pytz.UTC),
        time_taken = (t_end - t_start),
        error = error
    )
    json_obj = jsonable_encoder(res_obj)
    
    return JSONResponse(json_obj)

@app.get("/status")
async def status() -> JSONResponse:
    try:
      llm_status = llm.is_ready()
    #   llm_status = await llm.is_ready()
    except:
      llm_status = False

    try:
      encoder_status = encoder.is_ready()
    #   encoder_status = await encoder.is_ready()
    except:
      encoder_status = False

    try:
      translator_status = translator.is_ready()
    #   translator_status = await translator.is_ready()
    except:
      translator_status = False

    try:
      stt_status = stt.is_ready()
    except:
      stt_status = False

    try:
      tts_status = tts.is_ready()
    except:
      tts_status = False

    json_obj = {
        "llm": llm_status,
        "encoder": encoder_status,
        "translator": translator_status,
        "stt": stt_status,
        "tts": tts_status,
        "status": llm_status and encoder_status and translator_status and stt_status and tts_status
    }
    
    return JSONResponse(json_obj)
    

if __name__ == "__main__":
    host = os.getenv("TR_HOST")
    port = int(os.getenv("TR_PORT"))
    
    if os.getenv("EXPOSE_TO_PUBLIC"):
        if os.getenv("NGROK_TOKEN"):
            ngrok.set_auth_token(os.getenv("NGROK_TOKEN"))
        listener = ngrok.forward(port)

        print(f"[INFO] Public URL for {port}: {listener.url()}")
            
    print("[INFO] Compute Service started...")
    uvicorn.run(app, host=host, port=port)
