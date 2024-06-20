import os
import re
import bs4
import urllib.parse as urlparse
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.encoder import DocEmbeddings

class VectorDB:
    def __init__(self, data_path, chroma_db_path, embed_model_path):
        self.data_path = data_path
        self.chroma_db_path = chroma_db_path
        self.embeddings = DocEmbeddings()
        self.r_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.bs4_strainer = bs4.SoupStrainer() #class_=bs_classes)
        self.re_multiline = re.compile('\n+')
        
        self.create_db()
    
    @staticmethod
    def is_url(url):
        return urlparse.urlparse(url).scheme in ('http', 'https',)
        
    def post_process_documents(self, docs: list):
        new_docs = []
        
        for doc in docs:
            page_content = doc.page_content.strip()
            page_content = self.re_multiline.sub('\n', page_content)
            if page_content:
                doc.page_content = page_content
                new_docs.append(doc)
        
        return new_docs
    
    def pdf_processor(self, filepath):
        print(f"[INFO] Processing {filepath}...")
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        
        return docs
    
    def url_processor(self, url):
        print(f"[INFO] Processing {url}...")
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"parse_only": self.bs4_strainer},
        )
        docs = loader.load()
        
        return docs
    
    def chunk_documents(self, data_sources):
        documents = []
        for file in data_sources:
            if self.is_url(file):
                docs = self.url_processor(file)
            elif file.endswith('.pdf'):
                docs = self.pdf_processor(file)
            else:
                print(f"[WARNING] Skipping unsupported source document: {file}")
                continue
            
            documents.extend(docs)

        document_chunks = self.r_splitter.split_documents(documents)
        document_chunks = self.post_process_documents(document_chunks)
        
        return document_chunks
        
    def get_source_documents(self, data_source):
        if isinstance(data_source, list):
            return data_source
        elif os.path.isdir(data_source):
            source_documents = [os.path.join(self.data_path, x) for x in os.listdir(self.data_path)]
            return source_documents
        elif os.path.isfile(data_source):
            with open(data_source) as f:
                source_documents = [x.strip() for x in f.readlines()]
            return source_documents
    
    def create_db(self, db_name=None, data_source=None):
        if data_source is None:
            data_source = self.data_path
            
        if db_name is None:
            db_name = self.chroma_db_path
            
        source_documents = self.get_source_documents(data_source)
        document_chunks = self.chunk_documents(source_documents)

        vector_store = self.build_db_from_document_chunks(document_chunks, db_path=db_name)
        
        return vector_store
    
    def build_db_from_document_chunks(self, document_chunks, db_path):
        print("[INFO] Creating a new db...")
        vector_store = Chroma.from_documents(document_chunks, self.embeddings, persist_directory=db_path)
        
        return vector_store
    
    def add_to_db(self, data_source, db_name=None):
        if db_name is None:
            db_name = self.chroma_db_path
            
        if os.path.exists(db_name):
            vector_store = self.read_db(db_name)
        
        source_documents = self.get_source_documents(data_source)
        document_chunks = self.chunk_documents(source_documents)
        
        if os.path.exists(db_name):
            print("[INFO] Adding new documents to db...")
            vector_store.add_documents(document_chunks)
        else:
            vector_store = self.build_db_from_document_chunks(document_chunks, db_path=db_name)
        
        return vector_store
        
    
    def read_db(self, db_path):
        print("[INFO] Reading the db...")
        vector_store = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        
        return vector_store

    def get_retriever(self, db_name=None):
        if db_name is None:
            db_name = self.chroma_db_path
            
        if not os.path.exists(db_name):
            vector_store = self.create_db(db_name)
        else:
            vector_store = self.read_db(db_name)
            
        return vector_store.as_retriever()
    
    def clear_db(self, db_name):
        if db_name is None:
            db_name = self.chroma_db_path
            
        if os.path.exists(db_name):
            vector_store = self.read_db(db_name)
            print("[INFO] Clearining the db...")
            vector_store.delete_collection()
            
            return True
        
        return False
