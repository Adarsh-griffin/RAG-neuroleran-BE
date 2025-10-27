from langchain_qdrant.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import os

# Set local model path
local_model_path = "C:/techai/program/model/bge-base-en-v1.5"  # Adjust the path as needed

# Check if model exists, otherwise download
if not os.path.exists(local_model_path):
    model_name = "BAAI/bge-base-en-v1.5"
else:
    model_name = local_model_path

# Load Embeddings Once (match ingest.py settings exactly)
# BAAI/bge-base-en-v1.5 produces 768-dim vectors; keep normalize_embeddings=False to
# avoid mismatches with collections created during ingestion.
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Qdrant Connection
Qdrant_url = "http://localhost:6333" 

client = QdrantClient(url=Qdrant_url,  prefer_grpc=False)

# --- Start of Changed Section ---

# Function to Retrieve Documents
def get_relevant_docs(query, collection_name):
    # Initialize Qdrant vector store with the dynamic collection name
    db = Qdrant(client=client, embeddings=embeddings, collection_name=collection_name)

    # Create the retriever for the specific collection
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    return retriever

print("Retriever components initialized successfully!")

# --- End of Changed Section ---