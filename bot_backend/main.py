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

from src.db import VectorDB
from src.graph_app import GraphApp
from src.translator import Translator

load_dotenv()

os.makedirs(os.getenv("DB_BASE"), exist_ok=True)

app = FastAPI()
db = VectorDB(os.getenv("DB_DATA_PATH"), os.getenv("DB_PATH"))
graph_app = GraphApp()
translator = Translator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class Item(BaseModel):
    question: str
    session_hash: str
    src_lang: Optional[str]
    tgt_lang: Optional[str]
    context_only: Optional[bool] = True
    max_history: Optional[int] = 10
    db_path: Optional[str] = os.getenv("DB_PATH")


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
        success = True
    else:
        success = False
        error = "Encoder is not ready"
        
    return JSONResponse({"query": query, "content": content, "success": success, "error": error})

@app.post("/search_db")
async def search_db(db_path: str=Body(embed=True, default=None), search_query: dict=Body(embed=True)) -> JSONResponse:
    content = db.search_db(db_path, **search_query)
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
    
    if translator.is_ready():
        if item.src_lang == "sing":
            user_ip_en = translator.translate(item.question, src_lang="sing", tgt_lang="en")
        elif item.src_lang == "si":
            user_ip_en = translator.translate(item.question, src_lang="si", tgt_lang="en")
        else:
            user_ip_en = item.question
    else:
        success, error = False, "Translator is not ready"
        t_end = time.time()
        
        return {"answer": "", "si_answer": "", "success": success, "error": error, "time_taken": (t_end - t_start)}
        
    print(f"En query: {user_ip_en}")
    
    if graph_app.is_ready():
        answer = graph_app.chat(user_ip_en, session_id=item.session_hash, context_only=item.context_only, max_history=item.max_history, db_path=item.db_path)['messages'][-1]
        try:
            answer = json.loads(answer)["answer"]
        except Exception:
            ...
    else:
        success, error = False, "Bot is not ready"
        t_end = time.time()
        
        return {"answer": "", "si_answer": "", "success": success, "error": error, "time_taken": (t_end - t_start)}
    
    if translator.is_ready():
        if item.tgt_lang == "si":
            si_answer = translator.translate(answer, src_lang="en", tgt_lang="si")
        elif item.tgt_lang == "sing":
            si_answer = translator.translate(answer, src_lang="en", tgt_lang="sing")
        else:
            si_answer = answer
    else:
        success, error = False, "Translator is not ready"
        t_end = time.time()
        
        return {"answer": answer, "si_answer": "", "success": success, "error": error, "time_taken": (t_end - t_start)}
    
    print(f"En answer: {answer}")
    print(f"Tgt lang answer: {si_answer}")
    t_end = time.time()
    
    return {"answer": answer, "si_answer": si_answer, "success": True, "error": "", "time_taken": (t_end - t_start)}

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
