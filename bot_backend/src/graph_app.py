import re
import os
import json
import torch
import operator
import transformers
from string import Template
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


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str
    
    
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
    
    classifier_prompt = """You are given a question: "${question}". Your task is to 
    classify it into one of two categories: do_answer or do_rag. Not both 
    categories at once.

    do_answer: Questions that can be answered directly without referring to 
    any external documents or databases. These are typically general knowledge 
    questions, greetings, or common inquiries.

    do_rag: Questions that require referring to documents in a database to 
    provide a specific and accurate answer. These are usually detailed or 
    specific questions that need additional context or information from 
    external sources.

    Also provide an response for the questions without explanations how you 
    generated the answer in following format.

    (JSON object with the following keys)

    "category": "do_answer" or "do_rag"

    "answer": "The answer

    For question like "Hello", don't give the answer as "Hello is a common 
    greeting". Give a valid response similar to "Hi, How are you!"

    Just give output in a single JSON only in every time. Never provide two 
    JSONs in one output. nothing else. Specially don't provide two JSONs like 
    below in one output response.

    "category": "do_answer",
    "answer": "Hi, how can I assist you today?"

    "category": "do_rag",
    "answer": "To provide a specific answer, I would need to access relevant 
    documents or databases."

    """
    
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

        self.c_json = re.compile(r'\{.*?\}', re.DOTALL)
        
        self.chat_history = {}   
        
        self.app = self.get_langgraph() 

    # def chat(self, en_query, session_id) -> dict:
    def do_rag(self, state: AgentState) -> dict:
        session_id = state["session_id"]
        en_query = state["messages"][-2]
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
            
        result = self.rag_chain.invoke({"input": en_query, "chat_history": self.chat_history[session_id]})
        print(f">>> result: {result}")
        full_answer = result['answer']
        answer = full_answer.rsplit("### Response", 1)[-1].strip()

        self.chat_history[session_id].extend([HumanMessage(content=en_query), AIMessage(content=answer)])
        print("chat do_rag: ", self.chat_history[session_id])

        return {"messages": [answer], "session_id": state["session_id"]}
    
    def classifier(self, state: AgentState) -> dict:
        question = state["messages"][-1]
        prompt = Template(self.classifier_prompt)
        prompt = prompt.substitute({"question": question})
        messages = [
            {"role": "user", "content": prompt},
        ]

        output = self.llm.invoke(messages)
        response = self.c_json.findall(output)
        print(f"response: {response}")
        
        if response:
            response = response[0]
            classifier_response = json.loads(response)
            
            if classifier_response["category"] == "do_answer":
                self.chat_history[state["session_id"]].extend([HumanMessage(content=question), AIMessage(content=classifier_response["answer"])])
        else:
            response = '{"category": "do_rag", "answer": ""}'
                
        return {"messages": [response], "session_id": state["session_id"]}
    
    def where_to_go(self, state: AgentState) -> str:
        classifier_response = json.loads(state["messages"][-1])

        if classifier_response["category"] == "do_answer":
            return "answer"
        else:
            return "rag"
    
    def get_langgraph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("agent1", self.classifier)
        workflow.add_node("agent2", self.do_rag)

        workflow.add_conditional_edges(
            "agent1", self.where_to_go, {"answer": END, "rag": "agent2"}
        )
        workflow.add_edge("agent2", END)

        workflow.set_entry_point("agent1")

        app = workflow.compile()

        return app
    
    def chat(self, en_query, session_id):
        inputs = {"messages": [en_query], "session_id": session_id}
        answer = self.app.invoke(inputs)
        print(f"Answer: {answer} =====")
        
        return {"messages": answer["messages"]}
    
    def clear_history(self, session_id):
        if session_id in self.chat_history:
            self.chat_history[session_id] = []
