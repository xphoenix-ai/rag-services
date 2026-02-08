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
from utils.mapping.language_utils import get_language_maps

from src.llm.llm_base import LLMBase
from src.stt.stt_base import STTBase
from src.tts.tts_base import TTSBase
from src.encoder.encoder_base import EncoderBase
from src.translator.translator_base import TranslatorBase

load_dotenv()

description = """
This is the API Documentation of the Compute Service.

* Contains a translator, LLM, ASR, TTS and an Embedding service
* This is a state-less service
* Contains all the heavy operations of the system
* Independently run without other services
* Easily add new models by simpy adding inherited classes from the corresponding base classes
* The running models are set using the env file
"""

tags_metadata = [
    {
        "name": "root",
        "description": "Gives the basic info about the application.",
    },
    {
        "name": "languages",
        "description": "Gives all the supported languages by the compute service for each service (i.e. translator, tts, stt). The list of supported languages can vary depending on the selected models.",
    },
    {
        "name": "translate",
        "description": "Translates a given text from source language to a target language",
    },
    {
        "name": "generate",
        "description": "Generates LLM response to a given query.",
    },
    {
        "name": "transcribe",
        "description": "Transcribes an input audio array.",
    },
    {
        "name": "synthesize",
        "description": "Generates an audio array for a given text.",
    },
    {
        "name": "encode",
        "description": "Generates sentence embeddings for a given list of texts.",
    },
    {
        "name": "status",
        "description": "Gives the status of each service of the application.",
    },
]

app = FastAPI(
title="Compute Service",
    description=description,
    # summary="Compute service of the system",
    version="1.0",
    # terms_of_service="http://example.com/terms/",
    contact={
        "name": "xphoenix-ai",
        "url": "https://www.linkedin.com/company/xphoenix-ai",
        "email": "xphoenixai@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://github.com/xphoenix-ai/rag-services?tab=Apache-2.0-1-ov-file#readme",
    },
    openapi_tags=tags_metadata
)

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


llm: LLMBase = LLM(**llm_config["model_config"])
time.sleep(5)
stt: STTBase  = STT(**stt_config["model_config"])
time.sleep(10)
tts: TTSBase = TTS(**tts_config["model_config"])
time.sleep(10)
encoder: EncoderBase = Encoder(**encoder_config["model_config"])
time.sleep(20)
translator: TranslatorBase = Translator(**translator_config["model_config"])

translator_language_map, stt_language_map, tts_language_map = get_language_maps(translator_class, stt_class, tts_class)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class TrIntput(BaseModel):
    """Request model for translation endpoint.
    
    Attributes:
        src (str): Source text to translate
        src_lang (str): Source language code
        tgt_lang (str): Target language code
    """
    src: str
    src_lang: str
    tgt_lang: str
    


class TrOutput(BaseModel):
    """Response model for translation endpoint.
    
    Attributes:
        src (str): Original source text
        tgt (str): Translated text
        intermediate_res (str): Intermediate translation results if applicable
        src_lang (str): Source language code
        tgt_lang (str): Target language code
        modified_time (datetime): Timestamp of translation
        time_taken (float): Time taken for translation in seconds
        error (str): Error message if translation failed
    """
    src: str
    tgt: Optional[str]
    intermediate_res: Optional[str]
    src_lang: str
    tgt_lang: str
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]
        
    
class EmbdIntput(BaseModel):
    """Request model for embedding endpoint.
    
    Attributes:
        sentences (list[str]): List of sentences to encode
    """
    sentences: list[str]
    
    
class EmbdOutput(BaseModel):
    """Response model for embedding endpoint.
    
    Attributes:
        sentences (list[str]): Input sentences that were encoded
        embeddings (list[list[float]]): Embedding vectors for each sentence
        modified_time (datetime): Timestamp of encoding
        time_taken (float): Time taken for encoding in seconds
        error (str): Error message if encoding failed
    """
    sentences: list[str]
    embeddings: Optional[list[list[float]]]
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]
    

class LLMOutput(BaseModel):
    """Response model for LLM generation endpoint.
    
    Attributes:
        prompt (str): Input prompt sent to the model
        response (str): Generated text response
        modified_time (datetime): Timestamp of generation
        time_taken (float): Time taken for generation in seconds
        error (str): Error message if generation failed
    """
    prompt: str
    response: str
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]


class STTOutput(BaseModel):
    """Response model for speech-to-text endpoint.
    
    Attributes:
        transcription (str): Transcribed text from audio
        language (str): Detected language code
        modified_time (datetime): Timestamp of transcription
        time_taken (float): Time taken for transcription in seconds
        error (str): Error message if transcription failed
    """
    transcription: str
    language: str
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]


class TTSOutput(BaseModel):
    """Response model for text-to-speech endpoint.
    
    Attributes:
        audio_response (list): Audio samples as list
        sample_rate (int): Sample rate of the generated audio in Hz
        input_text (str): Text that was synthesized
        modified_time (datetime): Timestamp of synthesis
        time_taken (float): Time taken for synthesis in seconds
        error (str): Error message if synthesis failed
    """
    audio_response: list
    sample_rate: int
    input_text: str
    modified_time: Optional[datetime]
    time_taken: Optional[float]
    error: Optional[str]
    

@app.get("/", tags=["root"])
async def read_root() -> JSONResponse:
    """Get basic information about the Compute Service.
    
    Returns:
        JSONResponse: Service info with version and vendor details
    """
    return JSONResponse({"info": "Compute Service", "version": os.getenv("TR_VERSION"), "vendor": "XXX"})

@app.get("/languages", tags=["languages"])
async def get_supported_languages() -> JSONResponse:
    """Get list of supported languages for all services.
    
    Returns:
        JSONResponse: Dictionary with supported languages for translator, STT, and TTS
    """
    
    return JSONResponse({
       "translator": sorted([l.lower() for l in translator_language_map.keys()]),
       "stt": sorted([l.lower() for l in stt_language_map.keys()]),
       "tts": sorted([l.lower() for l in tts_language_map.keys()]), 
    })


@app.post("/translate", tags=["translate"])
async def translate(tr_item: TrIntput) -> JSONResponse:
    """Translate text from source language to target language.
    
    Args:
        tr_item (TrIntput): Translation request with source text and language codes
        
    Returns:
        JSONResponse: Translation result with metadata
    """
    t_start = time.time()
    
    src_lang = tr_item.src_lang
    tgt_lang = tr_item.tgt_lang
    src = tr_item.src

    try:
        res, error = translator.translate(src, src_lang.lower(), tgt_lang.lower())
    except Exception as e:
        res = ""
        error = f"{e}"

    t_end = time.time()
    
    res_obj = TrOutput(
    src = src,
    tgt = res,
    intermediate_res = "",
    src_lang = src_lang,
    tgt_lang = tgt_lang,
    modified_time = datetime.now(pytz.UTC),
    time_taken = (t_end - t_start),
    error = error
    )
    
    json_obj = jsonable_encoder(res_obj)
    
    return JSONResponse(json_obj)

@app.post("/encode", tags=["encode"])
async def encode(embed_item: EmbdIntput) -> JSONResponse:
    """Generate sentence embeddings for input texts.
    
    Args:
        embed_item (EmbdIntput): Request containing sentences to encode
        
    Returns:
        JSONResponse: Embeddings with metadata
    """
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

@app.post("/generate", tags=["generate"])
# async def generate(llm_input: LLMInput) -> JSONResponse:
async def generate(
    prompt: str=Body(embed=True), 
    generation_config: dict=Body(embed=True, default=llm_config["generation_config"])
    ) -> JSONResponse:
    """Generate text using the LLM model.
    
    Args:
        prompt (str): Input prompt for text generation
        generation_config (dict, optional): Model generation parameters
        
    Returns:
        JSONResponse: Generated text with metadata
    """
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

@app.post("/transcribe", tags=["transcribe"])
async def transcribe(audio_data: list=Body(embed=True), sample_rate: int=Body(embed=True, default=16_000), language: str=Body(embed=True)) -> JSONResponse:
    """Transcribe audio to text.
    
    Args:
        audio_data (list): Audio samples as list
        sample_rate (int, optional): Sample rate of audio in Hz. Defaults to 16000.
        language (str): Language code for transcription
        
    Returns:
        JSONResponse: Transcription result with detected language and metadata
    """
    t_start = time.time()

    audio_array = np.array(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    generation_config = stt_config.get("generation_config", {})
    transcription, language = stt.transcribe(audio_array, sample_rate, language, **generation_config)
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

@app.post("/synthesize", tags=["synthesize"])
async def synthesize(text: str=Body(embed=True), language: str=Body(embed=True, default="en")) -> JSONResponse:
    """Synthesize text to audio.
    
    Args:
        text (str): Text to synthesize into audio
        language (str, optional): Language code for synthesis. Defaults to "en".
        
    Returns:
        JSONResponse: Audio array with sample rate and metadata
    """
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

@app.get("/status", tags=["status"])
async def status() -> JSONResponse:
    """Get the status of all services (LLM, encoder, translator, STT, TTS).
    
    Returns:
        JSONResponse: Status of each service and overall service readiness
    """
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
