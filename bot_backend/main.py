import os
import sys
import pytz
import json
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel
from typing import Union, Optional
from dotenv import load_dotenv, dotenv_values
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from src.db import VectorDB
from src.graph_app import GraphApp
from src.translator import Translator

load_dotenv()

app = FastAPI()
db = VectorDB(os.getenv("DB_DATA_PATH"), os.getenv("DB_PATH"), os.getenv("EMBED_MODEL_PATH"))
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
    context_only: Optional[bool]
    max_history: Optional[int]


@app.get("/")
async def read_root() -> JSONResponse:
    return JSONResponse({"info": "DB Service", "version": os.getenv("DB_VERSION"), "vendor": "XXX"})

@app.post("/create")
def create_db() -> JSONResponse:
    db.create_db()
    
    return JSONResponse({"success": True})

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
    answer = graph_app.chat(user_ip_en, session_id=item.session_hash, context_only=item.context_only, max_history=item.max_history)['messages'][-1]
    
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
