import os
import sys
import time
import pytz
import json
import ngrok
import base64
import uvicorn
import nest_asyncio
# from pyngrok import ngrok, conf
from datetime import datetime
from pydantic import BaseModel
from typing import Union, Optional
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, dotenv_values
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Body

from langfuse import Langfuse

load_dotenv()

from src.stt import STT
from src.tts import TTS
from src.db import VectorDB
from src.graph_app import GraphApp
from src.translator import Translator


os.makedirs(os.getenv("DB_BASE"), exist_ok=True)

app = FastAPI()
db = VectorDB(os.getenv("DB_DATA_PATH"), os.getenv("DB_PATH"))
graph_app = GraphApp()
translator = Translator()
stt = STT()
tts = TTS()


tracing_enabled = os.getenv("TRACING_ENABLED")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST")

if tracing_enabled:
    langfuse = Langfuse(
      secret_key=langfuse_secret_key,
      public_key=langfuse_public_key,
      host=langfuse_host
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class Item(BaseModel):
    question: Optional[str] = ""
    audio_data: Optional[list] = []
    sample_rate: Optional[int] = 16_000
    session_hash: str
    src_lang: Optional[str]
    tgt_lang: Optional[str]
    context_only: Optional[bool] = True
    max_history: Optional[int] = 4
    db_path: Optional[str] = os.getenv("DB_PATH")
    free_chat_mode: Optional[bool] = False
    enable_audio_input: Optional[bool] = False
    enable_audio_output: Optional[bool] = False


def convert_first_letter(sentence):
    sentence = sentence.strip()
    words = sentence.split()
    if words[0].isupper():
        return sentence
    else:
        return sentence[0].lower() + sentence[1:]


@app.get("/")
async def read_root() -> JSONResponse:
    return JSONResponse({"info": "DB Service", "version": os.getenv("DB_VERSION"), "vendor": "XXX"})

@app.post("/create_db")
def create_db(db_path: str=Body(embed=True, default=None)) -> JSONResponse:
    print(f"db_path: {db_path} ===")
    
    error = ""
    if db.embeddings.is_ready():
        db.create_db(db_path=db_path)
        success = True
    else:
        success = False
        error = "Encoder is not ready"
    
    return JSONResponse({"success": success, "error": error})

@app.post("/update_file")
def update_db_from_file(file: UploadFile = File(...), db_path: str=Body(embed=True, default=os.getenv("DB_PATH"))) -> JSONResponse:
    os.makedirs(os.getenv("DB_DATA_PATH_UPLOAD"), exist_ok=True)
    
    content = file.file.read()
    file_path = os.path.join(os.getenv("DB_DATA_PATH_UPLOAD"), file.filename)
    
    with open(file_path, 'wb') as f:
        decoded_content = base64.b64decode(content)
        f.write(decoded_content)
    
    error = ""
    if db.embeddings.is_ready():        
        db.add_to_db(data_source=[file_path], db_path=db_path)
        graph_app.add_to_rag_chain_dict(db_path, force=True)
        success = True
    else:
        success = False
        error = "Encoder is not ready"
 
    return JSONResponse({"success": success, "error": error})

@app.post("/update_url")
def update_db_from_url(url: str=Body(embed=True), db_path: str=Body(embed=True, default=os.getenv("DB_PATH"))) -> JSONResponse:        
    error = ""
    if db.embeddings.is_ready():
        db.add_to_db(data_source=[url], db_path=db_path)
        graph_app.add_to_rag_chain_dict(db_path, force=True)
        success = True
    else:
        success = False
        error = "Encoder is not ready"
        
    return JSONResponse({"success": success, "error": error})

@app.post("/query_db")
async def query_db(query: str=Body(embed=True), db_path: str=Body(embed=True, default=None), k: int=Body(embed=True, default=4), 
                   return_score: int=Body(embed=True, default=True), return_relevance_socre: int=Body(embed=True, default=True),
                   search_kwargs: dict=Body(embed=True)) -> JSONResponse:
    error = ""
    if db.embeddings.is_ready():
        content = db.query_db(query, db_path=db_path, k=k, return_score=return_score, return_relevance_socre=return_relevance_socre, **search_kwargs)
        # content = await db.query_db(query, db_path=db_path, k=k, return_score=return_score, return_relevance_socre=return_relevance_socre, **search_kwargs)
        success = True
    else:
        success = False
        error = "Encoder is not ready"
        
    return JSONResponse({"query": query, "content": content, "success": success, "error": error})

@app.post("/search_db")
async def search_db(db_path: str=Body(embed=True, default=None), search_query: dict=Body(embed=True)) -> JSONResponse:
    content = db.search_db(db_path, **search_query)
    # content = await db.search_db(db_path, **search_query)
    
    return JSONResponse({"content": content})
        
@app.post("/clear_db")
def clear_db(db_path: str=Body(embed=True, default=None)) -> JSONResponse:
    print(f"db_name: {db_path} ===")
    graph_app.rag_chain = None
    success, error = db.clear_db(db_path)
    
    if success:
        graph_app.add_to_rag_chain_dict(db_path, force=True)
    
    return JSONResponse({"success": success, "error": error})

@app.post("/answer")
async def create_answer(item: Item) -> dict:
    t_start = time.time()
    print("session_hash ========> ", item.session_hash)
    print("src_lang ========> ", item.src_lang)
    print("tgt_lang ========> ", item.tgt_lang)
    print("max_history ========> ", item.max_history)
    
    # sample_rate, audio_data = None, []
    sample_rate, audio_data = None, None
    
    if item.enable_audio_input and not item.question and item.audio_data:
        if stt.is_ready():
            question, src_lang = stt.transcribe(item.audio_data, item.sample_rate, item.src_lang, item.tgt_lang)
            item.question = question
            item.src_lang = src_lang
            print(f"Transcription: {question}")
        else:
            success, error = False, "STT Service is not ready"
            t_end = time.time()
            
            return {"user_query": "", "en_answer": "", "answer": "", "success": success, "error": error, "time_taken": (t_end - t_start), "sample_rate": sample_rate, "audio_data": audio_data}
     
    if tracing_enabled:
        trace = langfuse.trace(
          name = "my-bot-trace",
          session_id = item.session_hash,
          metadata = {
            'source language':item.src_lang,
            'target language':item.tgt_lang
          },
          tags = ["production"],
          input = item.question,
        )
    else:
        trace = None        
    
    if item.src_lang == "en":
        user_ip_en = item.question
    elif translator.is_ready():
        if item.src_lang == "sing":
            question = convert_first_letter(item.question)
            user_ip_en = translator.translate(question, src_lang="sing", tgt_lang="en", trace_lf=trace)
            # user_ip_en = await translator.translate(item.question, src_lang="sing", tgt_lang="en")
        elif item.src_lang == "si":
            user_ip_en = translator.translate(item.question, src_lang="si", tgt_lang="en", trace_lf=trace)
            # user_ip_en = await translator.translate(item.question, src_lang="si", tgt_lang="en")
        else:
            user_ip_en = item.question
    else:
        success, error = False, "Translator is not ready"
        t_end = time.time()
        
        return {"user_query": item.question, "en_answer": "", "answer": "", "success": success, "error": error, "time_taken": (t_end - t_start), "sample_rate": sample_rate, "audio_data": audio_data}
        
    print(f"En query: {user_ip_en}")
    
    if graph_app.is_ready():
        en_answer = graph_app.chat(user_ip_en, 
                                   session_id=item.session_hash, 
                                   context_only=item.context_only, 
                                   max_history=item.max_history, 
                                   db_path=item.db_path, 
                                   free_chat_mode=item.free_chat_mode,
                                   trace_lf = trace
                                   )['messages'][-1]
        # en_answer = await graph_app.chat(user_ip_en, session_id=item.session_hash, context_only=item.context_only, max_history=item.max_history, db_path=item.db_path)['messages'][-1]
        try:
            en_answer = json.loads(en_answer)["answer"]
        except Exception:
            ...
    else:
        success, error = False, "Bot is not ready"
        t_end = time.time()
        
        return {"user_query": item.question, "en_answer": "", "answer": "", "success": success, "error": error, "time_taken": (t_end - t_start), "sample_rate": sample_rate, "audio_data": audio_data}
    
    if item.tgt_lang == "en":
        final_answer = en_answer
    elif translator.is_ready():
        # TODO: get rid of the following rules
        if item.tgt_lang == "si":
            final_answer = translator.translate(en_answer, src_lang="en", tgt_lang="si", trace_lf=trace)
            # final_answer = await translator.translate(en_answer, src_lang="en", tgt_lang="si")
            if "වාණිජ බැංකු" in final_answer:
                final_answer = final_answer.replace("වාණිජ බැංකු", "කොමර්ෂල් බැංකු")
        elif item.tgt_lang == "sing":
            final_answer = translator.translate(en_answer, src_lang="en", tgt_lang="sing", trace_lf=trace)
            # final_answer = await translator.translate(en_answer, src_lang="en", tgt_lang="sing")
            if "waanija benku" in final_answer:
                final_answer = final_answer.replace("waanija benku", "Commercial benku")
        else:
            final_answer = en_answer
    else:
        success, error = False, "Translator is not ready"
        
        if item.enable_audio_output and tts.is_ready():
            sample_rate, audio_data = tts.synthesize(en_answer, "en")

        t_end = time.time()
        
        return {"user_query": item.question, "en_answer": en_answer, "answer": "", "success": success, "error": error, "time_taken": (t_end - t_start), "sample_rate": sample_rate, "audio_data": audio_data}
    
    if item.enable_audio_output and tts.is_ready():
        # TODO: tts only supports for English at the moment
        sample_rate, audio_data = tts.synthesize(en_answer, "en")
    
    print(f"En answer: {en_answer}")
    print(f"Tgt lang answer: {final_answer}")
    t_end = time.time()
    
    if tracing_enabled:
        trace.update(output = final_answer)
    
    return {"user_query": item.question, "en_answer": en_answer, "answer": final_answer, "success": True, "error": "", "time_taken": (t_end - t_start), "sample_rate": sample_rate, "audio_data": audio_data}

@app.post("/clear_history")
async def clear_history(session_hash: str=Body(embed=True)) -> JSONResponse:
    success, error = graph_app.clear_history(session_hash)
    # success, error = await graph_app.clear_history(session_hash)
    
    return JSONResponse({"success": success, "error": error})

@app.get("/status")
async def status() -> JSONResponse:
    json_obj = {
        "llm": graph_app.llm.is_ready(),
        "translator": translator.is_ready(),
        "encoder": db.embeddings.is_ready(),
        "status": graph_app.llm.is_ready() and translator.is_ready() and db.embeddings.is_ready()
    }
    
    return JSONResponse(json_obj)

if __name__ == "__main__":
    host = os.getenv("APP_HOST")
    port = int(os.getenv("APP_PORT"))
    # compute_port = int(os.getenv("TR_PORT", "8001"))
    
    if os.getenv("EXPOSE_TO_PUBLIC"):
        if os.getenv("NGROK_TOKEN"):
            ngrok.set_auth_token(os.getenv("NGROK_TOKEN"))
        # listener_1 = ngrok.forward(compute_port)
        listener_2 = ngrok.forward(port)
        # ngrok_tunnel = ngrok.connect(port)

        # print(f"[INFO] Public URL: {ngrok_tunnel.public_url}")
        # print(f"[INFO] Public URL for {compute_port}: {listener_1.url()}")
        print(f"[INFO] Public URL for {port}: {listener_2.url()}")
        
    # nest_asyncio.apply()
    uvicorn.run(app, host=host, port=port)
    print("[INFO] Bot Service started...")
