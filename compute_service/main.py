import os
import sys
import pytz
import time
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel
from typing import Union, Optional
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, dotenv_values
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from src.llm import LLM
from src.encoder import Encoder
from src.translator import Translator

load_dotenv()

app = FastAPI()

llm = LLM(os.getenv("LLM_PATH"))
translator = Translator(os.getenv("TRANSLATOR_PATH"), os.getenv("TRANLITARATOR_PATH"))
encoder = Encoder(os.getenv("EMBED_MODEL_PATH"))

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
    

class LLMInput(BaseModel):
    prompt: str
    do_sample: bool
    max_new_tokens: int
    top_k: Optional[int]
    top_p: Optional[float]
    temperature: Optional[float]

class LLMOutput(BaseModel):
    prompt: str
    response: str
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]


@app.get("/")
async def read_root() -> JSONResponse:
    return JSONResponse({"info": "Translator Service", "version": os.getenv("TR_VERSION"), "vendor": "XXX"})

@app.post("/translate")
async def create_answer(tr_item: TrIntput) -> JSONResponse:
    t_start = time.time()
    
    src_lang = tr_item.src_lang
    tgt_lang = tr_item.tgt_lang
    src = tr_item.src
    error = ""
    
    if src_lang == "sing" and tgt_lang == "en":
        res = translator.singlish_to_english(src)
    elif src_lang == "sing" and tgt_lang == "si":
        res = translator.singlish_to_sinhala(src)
    elif src_lang == "en" and tgt_lang == "si":
        res = translator.english_to_sinhala(src)
    elif src_lang == "en" and tgt_lang == "sing":
        res = translator.english_to_singlish(src)
    elif src_lang == "si" and tgt_lang == "sing":
        res = translator.sinhala_to_singlish(src)
    elif src_lang == "si" and tgt_lang == "en":
        res = translator.sinhala_to_english(src)
    elif src_lang == tgt_lang:
        res = src
    else:
        res = ""
        error = f"Not supported language pair: {src_lang} and {tgt_lang}"
        # raise ValueError(f"Not supported language pair: {src_lang} and {tgt_lang}")
    
    t_end = time.time()
    res_obj = TrOutput(
        src = src,
        tgt = res,
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
async def generate(llm_input: LLMInput) -> JSONResponse:
    t_start = time.time()
    
    prompt = llm_input.prompt
    generation_config = llm_input.model_dump(exclude=["prompt"])
    response = llm.generate(prompt, **generation_config)
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
    

if __name__ == "__main__":
    host = os.getenv("TR_HOST")
    port = int(os.getenv("TR_PORT"))
    
    if os.getenv("EXPOSE_TO_PUBLIC"):
        if os.getenv("NGROK_TOKEN"):
            ngrok.set_auth_token(os.getenv("NGROK_TOKEN"))
        ngrok_tunnel = ngrok.connect(port)
        print(f"[INFO] Public URL: {ngrok_tunnel.public_url}")
    
    nest_asyncio.apply()
    uvicorn.run(app, host=host, port=port)
    print("[INFO] Translator service started...")
