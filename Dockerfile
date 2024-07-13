FROM continuumio/miniconda3

WORKDIR /rag_services

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate rag_env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure fastapi is installed:"
RUN python -c "import fastapi"

ADD /rag_services /rag_services

EXPOSE 8001
EXPOSE 8002
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# The code to run when container is started:
COPY entrypoint.sh ./
ENTRYPOINT ["./entrypoint.sh"]