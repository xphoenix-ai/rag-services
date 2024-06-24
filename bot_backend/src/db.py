import os
import re
import bs4
import urllib.parse as urlparse
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader, PyPDFLoader, TextLoader, RecursiveUrlLoader

from src.encoder import DocEmbeddings


class VectorDB:
    def __init__(self, data_path, chroma_db_path):
        self.data_path = data_path
        self.chroma_db_path = chroma_db_path
        self.embeddings = DocEmbeddings()
        self.r_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        self.bs4_strainer = bs4.SoupStrainer() #class_=bs_classes)
        self.re_multiline = re.compile('\n+')
        
        self.create_db()
        
    @staticmethod
    def bs4_extractor(html: str) -> str:
        soup = bs4.BeautifulSoup(html, "lxml")
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()

    @staticmethod
    def is_url(url):
        return urlparse.urlparse(url).scheme in ('http', 'https',)
        
    def __post_process_documents(self, docs: list):
        new_docs = []
        
        for doc in docs:
            page_content = doc.page_content.strip()
            page_content = self.re_multiline.sub('\n', page_content)
            
            if page_content:
                doc.page_content = page_content
                new_docs.append(doc)
        
        return new_docs
    
    def __txt_processor(self, filepath):
        print(f"[INFO] Processing {filepath}...")
        loader = TextLoader(filepath)
        docs = loader.load()
        
        return docs
    
    def __pdf_processor(self, filepath):
        print(f"[INFO] Processing {filepath}...")
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        
        return docs
    
    def __url_processor(self, url, max_depth=1):
        print(f"[INFO] Processing {url}...")
        loader = RecursiveUrlLoader(url, extractor=self.bs4_extractor, max_depth=max_depth)
        docs = loader.load()
        
        return docs
    
    def __url_processor_old(self, url):
        print(f"[INFO] Processing {url}...")
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"parse_only": self.bs4_strainer},
        )
        docs = loader.load()
        
        return docs
    
    def __chunk_documents(self, data_sources):
        documents = []
        for file in data_sources:
            if self.is_url(file):
                docs = self.__url_processor(file)
            elif file.endswith('.pdf'):
                docs = self.__pdf_processor(file)
            elif file.endswith('.txt'):
                docs = self.__txt_processor(file)
            else:
                print(f"[WARNING] Skipping unsupported source document: {file}")
                continue
            
            documents.extend(docs)

        document_chunks = self.r_splitter.split_documents(documents)
        document_chunks = self.__post_process_documents(document_chunks)
        
        return document_chunks
        
    def __get_source_documents(self, data_source):
        if isinstance(data_source, list):
            return data_source
        elif os.path.isdir(data_source):
            source_documents = [os.path.join(self.data_path, x) for x in os.listdir(self.data_path)]
            return source_documents
        elif os.path.isfile(data_source):
            with open(data_source) as f:
                source_documents = [x.strip() for x in f.readlines()]
            return source_documents
    
    def create_db(self, db_path=None, data_source=None):
        if data_source is None:
            data_source = self.data_path
            
        if db_path is None:
            db_path = self.chroma_db_path
            
        if not os.path.exists(db_path):
            source_documents = self.__get_source_documents(data_source)
            document_chunks = self.__chunk_documents(source_documents)
            vector_store = self.__build_db_from_document_chunks(document_chunks, db_path=db_path)
        else:
            vector_store = self.read_db(db_path)
            
        return vector_store
    
    def __build_db_from_document_chunks(self, document_chunks, db_path):
        if document_chunks:
            print(f"[INFO] Creating a new db at {db_path}...")
            vector_store = Chroma.from_documents(document_chunks, self.embeddings, persist_directory=db_path)
        else:
            print("[INFO] No documents found to build the db...")
        
        return vector_store
    
    def add_to_db(self, data_source, db_path=None):
        if db_path is None:
            db_path = self.chroma_db_path
            
        if os.path.exists(db_path):
            vector_store = self.read_db(db_path)
        
        source_documents = self.__get_source_documents(data_source)
        document_chunks = self.__chunk_documents(source_documents)
        
        if os.path.exists(db_path):
            if document_chunks:
                print("[INFO] Adding new documents to db...")
                vector_store.add_documents(document_chunks)
            else:
                print("[INFO] No documents found to add to db...")
        else:
            vector_store = self.__build_db_from_document_chunks(document_chunks, db_path=db_path)
        
        return vector_store
    
    def read_db(self, db_path):
        print(f"[INFO] Reading the db from {db_path}...")
        vector_store = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        
        return vector_store

    def get_retriever(self, db_path=None):
        if db_path is None:
            db_path = self.chroma_db_path
                        
        if not os.path.exists(db_path):
            vector_store = self.create_db(db_path)
        else:
            vector_store = self.read_db(db_path)
            
        return vector_store.as_retriever()
    
    def query_db(self, query, db_path=None, k=4):
        if db_path is None:
            db_path = self.chroma_db_path
                    
        db = self.read_db(db_path)
        docs = db.similarity_search(query, k=k)

        content = [doc.page_content for doc in docs]
        
        return content
    
    def clear_db(self, db_path):
        if db_path is None:
            print("[INFO] Specify the DB name to be cleared !")
            return False, "db_path is not specified"
            
        if os.path.exists(db_path):
            vector_store = self.read_db(db_path)
            print("[INFO] Clearining the db...")
            vector_store.delete_collection()
            
            return True, ""
        
        return False, f"{db_path} does not exist"
