# llm imports 
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# backend imports 
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn 
import asyncio 

# util imports 
from dotenv import load_dotenv
from typing import Optional
import os 
import chromadb 

# getting the environment variables 
load_dotenv()

# initializing the application 
app = FastAPI() 

# defining constants 
BASE_DIR: str = os.getcwd()  
DATA_DIR: str = "murli" 
VECTOR_DB_DIR: str = "murli_vector" 
CHUNK_SIZE: int = 512 
CHUNK_OVERLAP: int = 64
COLLECTION_NAME: str = "murliz"
MISTRAL_API_KEY: Optional[str] = os.environ.get("MISTRAL_API_KEY") 

# getting our default llms 
llm = MistralAI(api_key=MISTRAL_API_KEY) 
embed_model = HuggingFaceEmbedding() 

Settings.llm = llm 
Settings.embed_model = embed_model

# define custom schema 
class ChatModelSchema(BaseModel): 
    message : str 

# given a directory generates an index 
def get_index(data_directory: str, 
              chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> VectorStoreIndex:

    # getting the chroma client 
    client = chromadb.PersistentClient(VECTOR_DB_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME) 
    vector_store = ChromaVectorStore(chroma_collection=collection) 

    if VECTOR_DB_DIR not in os.listdir(BASE_DIR):
        reader = SimpleDirectoryReader(data_directory, recursive=True) 
        documents = reader.load_data()
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
        pipeline = IngestionPipeline(transformations=[text_splitter]) 
        nodes = pipeline.run(documents=documents) 

        # now we will do some patch up work by connecting them all 
        storage_context = StorageContext.from_defaults(vector_store=vector_store) 
        index = VectorStoreIndex(nodes, storage_context=storage_context) 
    else: 
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store) 
    return index 

index: VectorStoreIndex = get_index(DATA_DIR) 
query_engine = index.as_query_engine(similarity_top_k=10, streaming=True)

async def generate_response(message : str):
    response = query_engine.query(message) 
    for token in response.response_gen: 
        await asyncio.sleep(0.1) 
        yield token 

@app.post("/generate/") 
async def generate(chat: ChatModelSchema): 
    return StreamingResponse(
        generate_response(
            message=chat.message)
    ) 
 
if __name__ == "__main__":
    uvicorn.run(app, port=4000) 

