import os
import sys
import pytz
import json
import base64
import uvicorn
import nest_asyncio
from pyngrok import ngrok
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
    db.create_db(db_path=db_path)
    
    return JSONResponse({"success": True})

@app.post("/update_file")
def update_db_from_file(file: UploadFile = File(...), db_path: str=Body(embed=True, default=None)) -> JSONResponse:
    os.makedirs(os.getenv("DB_DATA_PATH_UPLOAD"), exist_ok=True)
    content = file.file.read()
    file_path = os.path.join(os.getenv("DB_DATA_PATH_UPLOAD"), file.filename)
    
    with open(file_path, 'wb') as f:
        decoded_content = base64.b64decode(content)
        f.write(decoded_content)
            
    db.add_to_db(data_source=[file_path], db_path=db_path)
 
    return JSONResponse({"success": True})

@app.post("/update_url")
def update_db_from_url(url: str=Body(embed=True), db_path: str=Body(embed=True, default=None)) -> JSONResponse:        
    db.add_to_db(data_source=[url], db_path=db_path)
 
    return JSONResponse({"success": True})

@app.post("/query_db")
async def query_db(query: str=Body(embed=True), db_path: str=Body(embed=True, default=None), k: int=Body(embed=True, default=4)) -> JSONResponse:
    content = db.query_db(query, db_path=db_path, k=k)
    
    return JSONResponse({"query": query, "content": content})

@app.post("/clear_db")
def clear_db(db_path: str=Body(embed=True, default=None)) -> JSONResponse:
    print(f"db_name: {db_path} ===")
    success, error = db.clear_db(db_path)
    
    return JSONResponse({"success": success, "error": error})

@app.post("/answer")
async def create_answer(item: Item) -> dict:
    print("session_hash ========> ", item.session_hash)
    print("src_lang ========> ", item.src_lang)
    print("tgt_lang ========> ", item.tgt_lang)
    print("max_history ========> ", item.max_history)
    
    if item.src_lang == "sing":
        user_ip_en = translator.translate(item.question, src_lang="sing", tgt_lang="en")
    elif item.src_lang == "si":
        user_ip_en = translator.translate(item.question, src_lang="si", tgt_lang="en")
    else:
        user_ip_en = item.question
        
    print(f"En query: {user_ip_en}")
    answer = graph_app.chat(user_ip_en, session_id=item.session_hash, context_only=item.context_only, max_history=item.max_history, db_path=item.db_path)['messages'][-1]
    
    try:
        answer = json.loads(answer)["answer"]
    except Exception:
        ...
    
    if item.tgt_lang == "si":
        si_answer = translator.translate(answer, src_lang="en", tgt_lang="si")
    elif item.tgt_lang == "sing":
        si_answer = translator.translate(answer, src_lang="en", tgt_lang="sing")
    else:
        si_answer = answer
    
    print(f"En answer: {answer}")
    print(f"Tgt lang answer: {si_answer}")
    
    return {"answer": answer, "si_answer": si_answer}

if __name__ == "__main__":
    host = os.getenv("APP_HOST")
    port = int(os.getenv("APP_PORT"))
    
    if os.getenv("EXPOSE_TO_PUBLIC"):
        if os.getenv("NGROK_TOKEN"):
            ngrok.set_auth_token(os.getenv("NGROK_TOKEN"))
        ngrok_tunnel = ngrok.connect(port)
        print(f"[INFO] Public URL: {ngrok_tunnel.public_url}")
        
    nest_asyncio.apply()
    uvicorn.run(app, host=host, port=port)
    print("[INFO] DB service started...")
