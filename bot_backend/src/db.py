import os
import re
import bs4
import time
import requests
import pandas as pd
from io import StringIO
import urllib.parse as urlparse
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader, PyPDFLoader, TextLoader, RecursiveUrlLoader, Docx2txtLoader, CSVLoader

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements, elements_to_json
from unstructured_client.utils import BackoffStrategy, RetryConfig

from src.encoder import DocEmbeddings


class VectorDB:
    
    c_unnamed = re.compile(r'\"Unnamed: \d+\":')

    def __init__(self, data_path, chroma_db_path):
        self.data_path = data_path
        self.chroma_db_path = chroma_db_path
        self.embeddings = DocEmbeddings()
        self.r_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        self.bs4_strainer = bs4.SoupStrainer() #class_=bs_classes)
        self.re_multiline = re.compile('\n+')

        self.pdf_loader_type = os.getenv("PDF_LOADER")
        self.api_key_auth = os.getenv("UNSTRUCTURED_API_KEY")
        self.unstructured_server_url = os.getenv("UNSTRUCTURED_SERVER_URL")
        # os.environ["DISABLE_NEST_ASYNCIO"] = "True"
        
        self.uns_client = None
        
        if self.pdf_loader_type == "Unstructured":
            if self.unstructured_server_url == "free-api":
                self.uns_client = UnstructuredClient(
                      api_key_auth= self.api_key_auth,
                      server=self.unstructured_server_url)
            else:
                self.uns_client = UnstructuredClient(
                      api_key_auth= self.api_key_auth,
                      server_url=self.unstructured_server_url)           
        
        # if self.embeddings.is_ready():
        #     self.create_db()
        # else:
        #     print("[INFO] Embedding service is not ready...")
            
        while not self.embeddings.is_ready():
            print("[INFO] Embedding service is not ready...")
            time.sleep(10)
        
        self.create_db()
        
    @staticmethod
    def bs4_extractor(html: str) -> str:
        soup = bs4.BeautifulSoup(html, "lxml")
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()
    
    @staticmethod
    def convert_html_content_to_text(content: str) -> str:
        soup = bs4.BeautifulSoup(content, 'lxml')
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()
    
    @staticmethod
    def url_data_extractor(url: str) -> Document:
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.content, 'html.parser')

        # Extract all tables and convert them to DataFrame strings
        tables = soup.find_all('table')
        for table in tables:
            # df = pd.read_html(str(table))[0]
            # df_string = df.to_string(index=False)
            
            df = pd.read_html(str(table))[0]
            df_string = df.to_json(orient = 'records')
            df_string = re.findall(r'\{(.*?)\}', df_string, re.DOTALL)
            df_string = "\n".join(df_string)
            df_string = VectorDB.c_unnamed.sub("", df_string)

            # Create a new paragraph tag with the DataFrame string
            new_tag = soup.new_tag('p')
            new_tag.string = df_string

            # Replace the table with the new paragraph tag
            table.replace_with(new_tag)
        
        html_text = VectorDB.convert_html_content_to_text(str(soup))
        doc = Document(page_content=html_text)
        
        return doc

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
    
    @staticmethod
    def extract_pdf_elements(file_path, client):
        with open(file_path, "rb") as f:
            files=shared.Files(
                content=f.read(),
                file_name=file_path,
            )

        req = shared.PartitionParameters(
            files=files,
            strategy="hi_res",
            hi_res_model_name="yolox",
            skip_infer_table_types=[],
            pdf_infer_table_structure=True,
        )

        try:
            resp = client.general.partition(req)
            time.sleep(10) 
            elements = dict_to_elements(resp.elements)
        except SDKError as e:
            print(e)

        return elements

    @staticmethod
    def get_table_html(table_element):
        table_html = table_element.metadata.text_as_html
        table_df = pd.read_html(StringIO(table_html))[0]

        return table_html, table_df
    
    @staticmethod
    def extract_doc_from_elements(pdf_elements):
        doclist = []
        doc = ""
        for row_idx in range(len(pdf_elements)):
            element_type = pdf_elements[row_idx].category
            element_content = pdf_elements[row_idx].text
            element_metadata = pdf_elements[row_idx].metadata
            next_element_type = pdf_elements[row_idx + 1].category if row_idx < len(pdf_elements) - 1 else 'None'

            if (element_type == 'Title' 
                and next_element_type != 'Title'
                and  doc != ""): # Don't add empty document
                doclist.append(Document(page_content=doc))
                dashline = "-" * len(element_content)
                doc = element_content + f"\n{dashline}" + "\n\n"
            elif (element_type == 'Title'):
                dashline = "-" * len(element_content)
                doc = doc + element_content + f"\n{dashline}" + "\n\n"
            elif (element_type == 'NarrativeText' 
                  or element_type == 'UncategorizedText'
                  or element_type == 'Image'):
                doc = doc + "\n\n" + element_content
            elif (element_type == 'Table'):
                _, df = VectorDB.get_table_html(pdf_elements[row_idx])
                df_string = df.to_json(orient = 'records')
                df_string = re.findall(r'\{(.*?)\}', df_string, re.DOTALL)
                table_str = "\n".join(df_string)  
                doc = doc + "\n\n" + table_str              

        doclist.append(Document(page_content=doc))
        return doclist

    def __csv_processor(self, filepath, csv_args={"delimiter": ","}):
        loader = CSVLoader(filepath, csv_args=csv_args)
        docs = loader.load()
        
        return docs
    
    def __docx_processor(self, filepath):
        print(f"[INFO] Processing {filepath}...")
        loader = Docx2txtLoader(filepath)
        docs = loader.load()
        
        return docs
    
    def __txt_processor(self, filepath):
        print(f"[INFO] Processing {filepath}...")
        loader = TextLoader(filepath)
        docs = loader.load()
        
        return docs
    
    def __pdf_processor(self, filepath):
        print(f"[INFO] Processing {filepath}...")
        
        if (self.pdf_loader_type == "PyPDF"):        
            loader = PyPDFLoader(filepath)
            docs = loader.load()
        elif (self.pdf_loader_type == "Unstructured"):
            pdf_elements = VectorDB.extract_pdf_elements(filepath, self.uns_client)
            docs = VectorDB.extract_doc_from_elements(pdf_elements)
        return docs
    
    def __url_processor_1(self, url, max_depth=1):
        print(f"[INFO] Processing {url}...")
        loader = RecursiveUrlLoader(url, extractor=self.bs4_extractor, max_depth=max_depth)
        docs = loader.load()
        
        return docs
    
    def __url_processor(self, url):
        print(f"[INFO] Processing {url}...")
        docs = self.url_data_extractor(url)
    
        return [docs]
    
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
            elif file.endswith('.docx'):    
                docs = self.__docx_processor(file)
            elif file.endswith('.csv'):
                docs = self.__csv_processor(file)   
            elif file.endswith('.tsv'):
                docs = self.__csv_processor(file, csv_args={"delimiter": "\t"})
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
        
    def __build_db_from_document_chunks(self, document_chunks, db_path):
        if document_chunks:
            print(f"[INFO] Creating a new db at {db_path}...")
            vector_store = Chroma.from_documents(document_chunks, self.embeddings, persist_directory=db_path)
        else:
            print("[INFO] No documents found to build the db...")
        
        return vector_store
    
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
            
        return vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1, "k": 5})
    
    def query_db(self, query, db_path=None, k=4, return_score=True, return_relevance_socre=False, **search_kwargs):
        if db_path is None:
            db_path = self.chroma_db_path
                    
        db = self.read_db(db_path)
        
        if return_score:
            if return_relevance_socre:
                docs = db.similarity_search_with_relevance_scores(query, k=k, **search_kwargs)
            else:
                docs = db.similarity_search_with_score(query, k=k)
            content = [(doc[1], doc[0].page_content) for doc in docs]
            
        else:
            docs = db.similarity_search(query, k=k)
            content = [doc.page_content for doc in docs]
        
        return content
    
    def search_db(self, db_path=None, **search_query):
        if db_path is None:
            db_path = self.chroma_db_path
                    
        db = self.read_db(db_path)
        results = db.get(**search_query)
        
        return results
    
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
