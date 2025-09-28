from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os 
from dotenv import load_dotenv 

#func to get api key from .env file
def get_fun(key_name):

    load_dotenv()
    try:
       key=os.getenv(key_name)
       return key
    except Exception as e:
         print(f"Error retrieving {key_name}: {e}")
         return None
    
def vectorstore_fun(chunks): 
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
 
    Qdrant_url = get_fun("Qdrant_url")
    Qdrant_api_key = get_fun("Qdrant_api_key") 
    collection_name = get_fun("collection_name")
    # INIT CLIENT 
    client = QdrantClient(
        url=Qdrant_url,
        api_key=Qdrant_api_key,
    )
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)   # 🔹 delete old collection
    # CREATE NEW COLLECTION
    vectorstore = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        collection_name=collection_name,
        url=Qdrant_url,
        api_key=Qdrant_api_key,
    )
    return vectorstore
