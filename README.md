# Rag Services - Any Language

 - An end-to-end RAG pipeline with both text and audio input-output support with fully customizable system architecture.
 - Supports cross lingual usage (any source language to any target language)
 - Pluggable modular architecture for any LLM, ASR, TTS, Embedding and Translation technology 

## System Overview
![Model](img/overall_system.jpeg)

## System Architecture
![Model](img/architecture.png)

## Demo Interface

https://github.com/user-attachments/assets/d1adec77-adcc-4673-8a26-7ba39de648e0

[//]: # (## Demo Interface)
[//]: # (![Model]&#40;img/sample_chat.png&#41;)

[**Watch our Demo Video**](https://drive.google.com/file/d/1yqi3q2ZIxqeI7gozgqBCAk5PSUeSyaBv/view?usp=sharing)

## Components
### Compute Service:
This is the heavy computation services of the system.
- **LLM Service** - LLM is up and running here
- **Embedding Service** - Sentence/Document embedding service is running here
- **Translator Service** - All direction translation service is running here
- **STT Service** - Speech-To-Text service is running here
- **TTS Service** - Text-To-Speech service is running here

### Bot Backend:
This is the full RAG pipeline which answers a user query using the available knowledge bases fed to the system.

- **Bot Service** - The RAG pipeline
- **DB Service** - The RAG knowledge base store
### Client App:
Client frontend App that the user interacts with the bot/system.

## Getting Started

### Setup the Environment

  * You can create the conda environment named `rag_env` with the given `environment.yml` file.
  * **Note:**
    * `environment.yml` builds the environment without the GPU usage.
    * For GPU usage,
      * First, you may use `environment.yml` and build the environment
      * Then install Pytorch (and other GPU packages if any) with CUDA support using conda (or pip) separately in the environment again. 
  ```shell
    conda env create -f environment.yml
  ```

### Start the System

The 3 services should be run as 3 separate services (in separate terminals).
- Compute Service is independent of others
- Bot Backend is depending upon the Compute Service
- Bot Frontend is depending upon the Bot Backend

You can access the services as follows

1. Start the compute service

Check .env file and the yml files of each service. You may need to fill certain fields in yml files. In .env file, keep fields empty if a variable should be set as `False`.
```
conda activate rag_env
cd compute_service
python main.py
```

2. Start the bot backend

- Check the .env file. Keep fields empty if a variable should be set as `False`.
- For advanced PDF processing (i.e. table data extraction) we recommend to use [unstructured-API](https://github.com/Unstructured-IO/unstructured-api), i.e. PDF_LOADER="Unstructured" in .env (defaults to "PyPDF")
```
conda activate rag_env
cd bot_backend
python main.py
```

3. Start the frontend app
```
conda activate rag_env
cd bot_frontend
python app_v2.py
```


## Dockerization

The services can be containerized using the following steps.
### Build the Image:
```docker build -t rag_services .```

### Run the Container
```docker run --gpus all -p 8001:8001 -p 8002:8002 -p 7860:7860 rag_services```

You can access the services as follws
#### Linux:
- compute service: http://127.0.0.1:8001
- bot backend: http://127.0.0.1:8002
- client app: http://127.0.0.1:7860

#### Windows (127.0.0.1 may not work in Windows):
- compute service: http://host.docker.internal:8001
- bot backend: http://host.docker.internal:8002
- client app: http://host.docker.internal:7860

## Roadmap

- [ ] Complete Bot Backend
    - [x] Basic RAG Flow
    - [x] Session Management
    - [x] RAG mode and LLM-only chat mode
    - [x] Handle both text and voice input and output
    - [x] Add knowledge to vector db through API
    - [x] Trace Responses
    - [ ] Tool Calling
    - [ ] Further Improvements
- [ ] Complete Compute Service
    - [ ] LLM Service
        - [x] [Huggingface](https://huggingface.co/)
        - [x] [Ollama](https://ollama.com/)
        - [x] [Llama-cpp](https://github.com/ggerganov/llama.cpp)
        - [ ] [Openai](https://openai.com/api/)
    - [ ] Embedding Service
        - [x] Huggingface
        - [x] [Sentence Transformers](https://sbert.net/)
        - [ ] Openai
    - [ ] Translation Service
        - [x] Huggingface
        - [ ] [Google Translate API](https://cloud.google.com/translate/docs/reference/rest)
    - [ ] ASR Service
        - [x] Huggingface
        - [x] [Openai-whisper](https://github.com/openai/whisper)
    - [ ] TTS Service
        - [x] Huggingface
        - [x] [CoquiTTS](https://github.com/coqui-ai/TTS)
- [ ] Complete Frontend APP
    - [x] Basic chat interface
    - [x] Add knowledge to RAG (i.e. File Upload, URL fetch)
    - [ ] Get rid of Gradio
- [ ] Update Docker Image
- [x] Generalize Multilingual Support
- [ ] Voice Streaming Capability

## Contributors

- [Kasun Wickramasinghe](https://www.linkedin.com/in/kasun-wickramasinghe-7b746a152/)
- [Sachin Siriwardhana](https://www.linkedin.com/in/sachinsiriwardhana/)
- [Nipuna Solangaarachchi](https://www.linkedin.com/in/nipuna-solangaarachchi-00136b15b/)

## Contact Us

[xphoenixai@gmail.com](mailto:xphoenixai@gmail.com)

## Social Media

Follow our social media channels for latest updates

<a href="https://www.linkedin.com/company/xphoenix-ai">
    <img src="img/Linkedin_icon.png" alt="Version" width="50" height="50">
</a>
<a href="https://web.facebook.com/profile.php?id=61571067352559">
    <img src="img/Facebook_icon.png" alt="Version" width="50" height="50">
</a>
