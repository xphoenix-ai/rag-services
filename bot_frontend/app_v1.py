import json
import ngrok
import codecs
import base64
import requests
import gradio as gr
from PIL import Image
from typing import Any
 
# URL = "http://127.0.0.1:8000/"
URL = "http://localhost:8002"

LANGUAGE_MAPPING = {'Singlish':'sing', 'English':'en', 'Sinhala':'si'}
 
def add_text(history, text: str):
    if not text:
        raise gr.Error('Please enter text to start the conversation')
    #history = history + [({"role": "user", "content": text}, {})]
    history = history + [(text,'')]
    return history
 
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
 
def upload_file(files):
 
    url = f"{URL}/update_file"
    file_paths = [file.name for file in files]
    print(file_paths)
 
    for file in files:
        with open(file, 'rb') as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
            files = {'file': (file.name, base64_pdf, 'application/pdf')}
            data_json = {'db_path': 'dbpath'}
           
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
          with gr.Row():          
              chatbot = gr.Chatbot(value=[], elem_id='chatbot',avatar_images=["icons/me.jpg", "icons/combank.jpg"])
          with gr.Row():
                txt = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press enter",
                          )
          with gr.Row():
                submit_btn = gr.Button('submit')
                clear_btn = gr.Button('clear')
       
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
 
    submit_btn.click(
            fn=add_text,
            inputs=[chatbot,txt],
            outputs=[chatbot, ],
            queue=True).success(
            fn=generate_answer,
            inputs = [chatbot, txt, src_language, tgt_language],
            outputs = [chatbot,txt])
    
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
