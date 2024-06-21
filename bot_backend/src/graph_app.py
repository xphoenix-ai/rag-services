import os
import json
import torch
import operator
import transformers
from langchain import HuggingFacePipeline
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, ConversationalRetrievalChain

from src.db import VectorDB
from src.llm import CustomLLM


class GraphApp:
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer in ONE sentence. DON'T give notes (Note:) within brackets.\
    DO NOT build sample dialogues like \nUser: bla bla. \nAI: bla bla bla

    {context}"""
    
    def __init__(self):
        # self.llm = CustomLLM(os.getenv("LLM_URL"))
        self.llm = CustomLLM()
        
        vector_db = VectorDB(os.getenv("DB_DATA_PATH"), os.getenv("DB_PATH"), os.getenv("EMBED_MODEL_PATH"))
        retriever = vector_db.get_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        self.chat_history = {}    

    def chat(self, en_query, session_id) -> dict:
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
            
        result = self.rag_chain.invoke({"input": en_query, "chat_history": self.chat_history[session_id]})
        print(f">>> result: {result}")
        full_answer = result['answer']
        answer = full_answer.rsplit("### Response", 1)[-1].strip()

        self.chat_history[session_id].extend([HumanMessage(content=en_query), AIMessage(content=answer)])
        print("chat do_rag: ", self.chat_history[session_id])

        return {"messages": [answer]}
    
    def clear_history(self, session_id):
        if session_id in self.chat_history:
            self.chat_history[session_id] = []
