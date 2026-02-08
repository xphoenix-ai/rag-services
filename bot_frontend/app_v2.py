import os
import time
import json
import ngrok
import codecs
import base64
import requests
import numpy as np
import gradio as gr
from PIL import Image
from typing import Any
 
# URL = "http://127.0.0.1:8000/"
URL = "http://127.0.0.1:8002"


def add_text(history, text: str):
    """Add user text input to conversation history.
    
    Args:
        history (list): Current conversation history
        text (str): User input text
        
    Returns:
        list: Updated conversation history
        
    Raises:
        gr.Error: If text is empty
    """
    if not text:
        raise gr.Error('Please enter text to start the conversation')
    #history = history + [({"role": "user", "content": text}, {})]
    history = history + [(text,'')]
    return history

def check_input(audio_input, txt_input):
    """Validate that user provided either audio or text input.
    
    Args:
        audio_input: Audio recording from microphone
        txt_input (str): Text input from user
        
    Raises:
        gr.Error: If neither audio nor text input is provided
    """
    try:
        sr, audio_array = audio_input
    except:
        audio_array = np.array([])
    
    if audio_array.size == 0 and (not txt_input):
        raise gr.Error('Please say or type something to proceed')

def reverse_audio(audio, reverse=False):
    """Reverse or keep audio data unchanged.
    
    Args:
        audio: Tuple of (sample_rate, audio_array)
        reverse (bool, optional): Whether to reverse the audio. Defaults to False.
        
    Returns:
        Tuple[int, np.ndarray]: (sample_rate, audio_array) unchanged or reversed
    """
    sr, data = audio
    return (sr, np.flipud(data) if reverse else data)
 
def upload_website(text: str):
    """Upload content from a website URL to the knowledge base.
    
    Args:
        text (str): URL to upload from
        
    Displays success or error message based on API response.
    """
 
    url = f"{URL}/update_url"
    data_json = {'url': text}
   
    response = requests.post(url, json=data_json)
    if response.ok:
        gr.Info(f"Data from {text} url been uploaded successfully")
        #return ''
    gr.Error(f"Failed to upload {text} details. Please try again. Status code: {response.status_code}")
    #return ''
 
def generate_answer(history, text: str, src_language:str, tgt_language:str, request: gr.Request):
    """Generate an answer to a user query with language translation.
    
    Args:
        history (list): Conversation history
        text (str): User query
        src_language (str): Source language code
        tgt_language (str): Target language code
        request (gr.Request): Gradio request object for session handling
        
    Yields:
        Tuple[list, str]: Updated history and empty input field
    """
   
    url = f"{URL}/answer"
 
    if txt:
        data_dict = {
            "question": text,
            "session_hash": request.session_hash,
            "src_lang": src_language,
            "tgt_lang": tgt_language
        }
        response = requests.post(url, json=data_dict)
        response_dict = json.loads(response.content)
    
        for char in str(response_dict['si_answer']):  
            if history:
                history[-1][-1] += char
            yield history,''

def stream_answer(history, answer_txt):
    """Stream answer text character by character to conversation history.
    
    Args:
        history (list): Conversation history
        answer_txt (str): Answer text to stream
        
    Yields:
        list: Updated history with progressively more answer text
    """
    # global answer_txt
    # answer = history[-1][-1]
    history[-1][-1] = ''

    for char in answer_txt:
        time.sleep(0.01)
        if history:
            history[-1][-1] += char
        yield history
            
def generate_audio_answer(history: list, text: str, audio_input: np.ndarray, src_language:str, tgt_language:str, free_chat_mode: bool, enable_audio_input: bool, enable_audio_output: bool, request: gr.Request):
    """Generate answer with audio input and/or output support.
    
    Args:
        history (list): Conversation history
        text (str): Text query
        audio_input (np.ndarray): Audio input from microphone
        src_language (str): Source language code
        tgt_language (str): Target language code
        free_chat_mode (bool): Whether to use free chat without RAG
        enable_audio_input (bool): Whether to process audio input
        enable_audio_output (bool): Whether to generate audio output
        request (gr.Request): Gradio request object for session handling
        
    Yields:
        Tuple[list, str, Tuple]: Updated history, empty input, and optional audio output
    """
    # global answer_txt
    url = f"{URL}/answer"
    
    if audio_input:
        sr, audio_array = audio_input
    else:
        sr, audio_array = 16_000, np.array([])
        
    print(f"sr: {sr}")
    print(f"audio_arry: {audio_array.shape}")
    assert sr == 16_000, f"Input Audio sample rate should be 16000Hz but got {sr}Hz"
    
    data_dict = {
        "session_hash": request.session_hash,
        "question": text,
        "audio_data": audio_array.tolist(),
        "sample_rate": sr,
        "src_lang": src_language,
        "tgt_lang": tgt_language,
        "free_chat_mode": free_chat_mode,
        "enable_audio_input": enable_audio_input,
        "enable_audio_output": enable_audio_output
    }
    response = requests.post(url, json=data_dict)
    response_dict = json.loads(response.content)
    
    # print(f"response: {response_dict}")

    user_query = response_dict['user_query']
    answer_txt = response_dict['answer']
    sr = response_dict['sample_rate']
    audio_response = np.array(response_dict['audio_data'])
    
    history.append((user_query, ""))

    return history, "", None, (sr, audio_response) if enable_audio_output else None, answer_txt

def upload_file(files):
 
    url = f"{URL}/update_file"
    file_paths = [file.name for file in files]
    print(file_paths)
 
    for file in files:
        with open(file, 'rb') as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            files = {'file': (os.path.basename(file.name), base64_pdf, 'application/pdf')}
            data_json = {}
           
            #response = requests.post(url, files=files, data=data_json)
            response = requests.post(url, files=files, data=data_json)
            if response.json()["success"]:
                gr.Info(file.name.replace('\\', '/').split('/')[-1] +" uploaded successfully")
            else:
                gr.Warning("Failed to upload the file. Please try again.")
                

def clear_history(history, request: gr.Request):
    url = f"{URL}/clear_history"
    body = {'session_hash': request.session_hash}
    response = requests.post(url, json=body)
    if response.json()["success"]:
        return ''
    else:
        gr.Warning("Failed to clear history.")
        # return history
        return ''
    
    
def change_audio(enable_audio_checkbox, enable_audio_input, enable_audio_output):
    src_lang_list, tgt_lang_list = get_supported_languages(enable_audio_input, enable_audio_output)

    if enable_audio_checkbox:
        return gr.Audio(visible=True), gr.Dropdown(choices=src_lang_list), gr.Dropdown(choices=tgt_lang_list)
    else:
        return gr.Audio(visible=False), gr.Dropdown(choices=src_lang_list), gr.Dropdown(choices=tgt_lang_list)


def common_languages(lang_list_1, lang_list_2, sort=True):
    lang_list_1 = set(lang_list_1)
    lang_list_2 = set(lang_list_2)
    common = list(lang_list_1.intersection(lang_list_2))

    return sorted(common) if sort else common


def get_supported_languages(enable_audio_input, enable_audio_output):
    url = f"{URL}/languages"
    response = requests.get(url)
    response = response.json()

    src_lang_list = []
    tgt_lang_list = []

    if response["success"]:
        all_languages = response["languages"]
        tr_languages = all_languages["translator"]
        stt_languages = all_languages["stt"]
        tts_languages = all_languages["tts"]

        if enable_audio_input:
            src_lang_list = common_languages(tr_languages, stt_languages)
        else:
            src_lang_list = tr_languages

        if enable_audio_output:
            tgt_lang_list = common_languages(tr_languages, tts_languages)
        else:
            tgt_lang_list = tr_languages

    return src_lang_list, tgt_lang_list


init_src_lang_list, init_tgt_lang_list = get_supported_languages(False, False)


with gr.Blocks() as demo:
    answer_txt = gr.State([])
    
    with gr.Tab("Chatbot"):
      with gr.Column():
          with gr.Row():  
              src_language = gr.Dropdown(
                  init_src_lang_list, label="Select Source Language", value="english", allow_custom_value=True
              )
              tgt_language = gr.Dropdown(
                  init_tgt_lang_list, label="Select Target Language", value="english", allow_custom_value=True
              )
              with gr.Column():
                free_chat_mode = gr.Checkbox(label="Free Chat Mode")
                enable_audio_input = gr.Checkbox(label="Enable Audio Input")
                enable_audio_output = gr.Checkbox(label="Enable Audio Output")
      with gr.Column():
        with gr.Row():
            audio_input_block = gr.Audio(
                sources=["microphone"],
                waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066B4",
                    skip_length=2,
                    show_controls=False,
                    sample_rate=16_000
                ),
                label="audio_input",
                visible=False
            )
        with gr.Row():
            txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                    )
        with gr.Row():
            submit_btn = gr.Button('submit')
            clear_btn = gr.Button('clear')
            
        with gr.Row():          
            chatbot = gr.Chatbot(value=[], elem_id='chatbot',avatar_images=["icons/me.jpg", "icons/bot.png"], type="tuples")
        with gr.Row():
            audio_output_block = gr.Audio(autoplay=True, label="audio_output", visible=False)
       
    with gr.Tab("Advanced"):
        with gr.Row():
            with gr.Column():
              file_output = gr.File()
            with gr.Column():
              upload_button = gr.UploadButton("📁 Upload documents", file_types=[".pdf"], file_count="multiple")
             
        with gr.Row():
            with gr.Column():  
              web_url = gr.Textbox(
                  show_label=False,
                  placeholder="Enter webpage url",
                  )
 
            with gr.Column():
                url_submit_btn = gr.Button('Submit URL')

    upload_button.upload(upload_file, upload_button, file_output)
        
    submit_btn.click(
        fn=check_input,
        inputs=[audio_input_block, txt],
        outputs=[],
        queue=True
    ).success(
        fn=generate_audio_answer,
        inputs = [chatbot, txt, audio_input_block, src_language, tgt_language, free_chat_mode, enable_audio_input, enable_audio_output],
        outputs = [chatbot, txt, audio_input_block, audio_output_block, answer_txt]
    ).success(
        fn=stream_answer,
        inputs = [chatbot, answer_txt],
        outputs = [chatbot]
    )

    clear_btn.click(
        fn=clear_history,
        inputs=[chatbot],
        outputs=[chatbot],
        queue=True)
 
    url_submit_btn.click(
            fn=upload_website,
            inputs=[web_url],
            outputs=[web_url],
            queue=True)
    
    enable_audio_input.change(fn=change_audio, inputs=[enable_audio_input, enable_audio_input, enable_audio_output], outputs=[audio_input_block, src_language, tgt_language])
    enable_audio_output.change(fn=change_audio, inputs=[enable_audio_output, enable_audio_input, enable_audio_output], outputs=[audio_output_block, src_language, tgt_language])

demo.queue()
# demo.launch(debug=True, server_port=8003)
demo.launch(debug=True)
