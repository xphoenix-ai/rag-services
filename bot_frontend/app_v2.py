import os
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

LANGUAGE_MAPPING = {'Singlish':'sing', 'English':'en', 'Sinhala':'si'}
 
def add_text(history, text: str):
    if not text:
        raise gr.Error('Please enter text to start the conversation')
    #history = history + [({"role": "user", "content": text}, {})]
    history = history + [(text,'')]
    return history

def check_input(audio_input, txt_input):
    try:
        sr, audio_array = audio_input
    except:
        audio_array = np.array([])
    
    if audio_array.size == 0 and (not txt_input):
        raise gr.Error('Please say or type something to proceed')

def reverse_audio(audio, reverse=False):
    sr, data = audio
    return (sr, np.flipud(data) if reverse else data)
 
def upload_website(text: str):
 
    url = f"{URL}/update_url"
    data_json = {'url': text}
   
    response = requests.post(url, json=data_json)
    if response.ok:
        gr.Info(f"Data from {text} url been uploaded successfully")
        #return ''
    gr.Error(f"Failed to upload {text} details. Please try again. Status code: {response.status_code}")
    #return ''
 
def generate_answer(history, text: str, src_language:str, tgt_language:str, request: gr.Request):
   
    url = f"{URL}/answer"
 
    src_ = LANGUAGE_MAPPING[src_language]
    tgt_ = LANGUAGE_MAPPING[tgt_language]
 
    if txt:
        data_dict = {
            "question": text,
            "session_hash": request.session_hash,
            "src_lang": src_,
            "tgt_lang": tgt_
        }
        response = requests.post(url, json=data_dict)
        response_dict = json.loads(response.content)
    
        for char in str(response_dict['si_answer']):  
            if history:
                history[-1][-1] += char
            yield history,''
            
def generate_audio_answer(history: list, text: str, audio_input: np.ndarray, src_language:str, tgt_language:str, free_chat_mode: bool, enable_audio_input: bool, enable_audio_output: bool, request: gr.Request):
    url = f"{URL}/answer"
 
    src_ = LANGUAGE_MAPPING[src_language]
    tgt_ = LANGUAGE_MAPPING[tgt_language]
    
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
        "src_lang": src_,
        "tgt_lang": tgt_,
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
    
    history.append((user_query, answer_txt))
    
    return history, "", None, (sr, audio_response) if enable_audio_output else None
 
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
    
    
def change_audio(enable_audio_checkbox):
    if enable_audio_checkbox:
        return gr.Audio(visible=True) #make it visible
    else:
        return gr.Audio(visible=False)
 
 
with gr.Blocks() as demo:
    with gr.Tab("Chatbot"):
      with gr.Column():
          with gr.Row():  
              src_language = gr.Dropdown(
                  ["Singlish", "Sinhala", "English"], label="Select Source Language", value="Singlish"
              )
              tgt_language = gr.Dropdown(
                  ["Singlish", "Sinhala", "English"], label="Select Target Language", value="Singlish"
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
            chatbot = gr.Chatbot(value=[], elem_id='chatbot',avatar_images=["icons/me.jpg", "icons/combank.jpg"])
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
 
    # btn.upload(
    #         fn=render_first,
    #         inputs=[btn],
    #         outputs=[chatbot],)
 
    upload_button.upload(upload_file, upload_button, file_output)
 
    # submit_btn.click(
    #     fn=add_text,
    #     inputs=[chatbot, txt],
    #     outputs=[chatbot, ],
    #     queue=True).success(
    #     fn=generate_answer,
    #     inputs = [chatbot, txt, src_language, tgt_language],
    #     outputs = [chatbot,txt]
    # )
        
    submit_btn.click(
        fn=check_input,
        inputs=[audio_input_block, txt],
        outputs=[],
        queue=True
    ).success(
        fn=generate_audio_answer,
        inputs = [chatbot, txt, audio_input_block, src_language, tgt_language, free_chat_mode, enable_audio_input, enable_audio_output],
        outputs = [chatbot, txt, audio_input_block, audio_output_block]
    )
    
    # submit_btn.click(
    #     fn=reverse_audio,
    #     inputs=[audio_input_block],
    #     outputs=[audio_output_block],
    #     queue=True
    # )    

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
    
    enable_audio_input.change(fn=change_audio, inputs=[enable_audio_input], outputs=[audio_input_block])
    enable_audio_output.change(fn=change_audio, inputs=[enable_audio_output], outputs=[audio_output_block])

demo.queue()
# demo.launch(debug=True, server_port=8003)
demo.launch(debug=True)

# ngrok.set_auth_token("2i0ojfXL9dDFQZaLnQYEByfuzUL_6fqw9pkF2NTKos6hvgbhr")
# ngrok.set_auth_token("2iZKHcWVWgZvPjIlcTzH8D2NteJ_2tEPydcco5cfZfjkkqJDT")
# listener_1 = ngrok.forward(8001)
# listener_2 = ngrok.forward(8002)
# listener_3 = ngrok.forward(7860)
# print(f"[INFO] Public URL for {8001}: {listener_1.url()}") 
# print(f"[INFO] Public URL for {8002}: {listener_2.url()}") 
# print(f"[INFO] Public URL for {7860}: {listener_3.url()}") 
