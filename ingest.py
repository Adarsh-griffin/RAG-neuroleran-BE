
import sys
import os
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest_document(file_path):
    """
    Loads a PDF, splits it into chunks, creates embeddings,
    and stores them in a Qdrant collection.
    The collection name is derived from the PDF's filename.
    """
    print(f"🚀 Starting ingestion process for: {file_path}")

    try:
        # --- 1. Load and split the PDF document ---
        loader = PyPDFLoader(file_path) # Dynamically use the provided file path
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)
        print(f"📄 Loaded and split {len(documents)} document pages into {len(texts)} chunks.")

        # --- 2. Load the HuggingFace embedding model ---
        model = "BAAI/bge-base-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("✅ Embedding model loaded.")

        # --- 3. Configure Qdrant and create a dynamic collection name ---
        qdrant_cloud_url = "http://localhost:6333"
        
        # Create a collection name from the filename (e.g., "my_document.pdf" -> "my_document")
        base_filename = os.path.basename(file_path)
        collection_name = os.path.splitext(base_filename)[0].lower().replace(" ", "_")
        
        print(f"🎯 Using Qdrant collection: '{collection_name}'")

        # --- 4. Create or connect to the Qdrant collection ---
        Qdrant.from_documents(
            texts,
            embeddings,
            url=qdrant_cloud_url,
            collection_name=collection_name,
            prefer_grpc=False
        )

        print(f"✅ Vector database updated in Qdrant for collection '{collection_name}'.")

    except Exception as e:
        print(f"❌ An error occurred during ingestion: {e}")

# This block allows the script to be run from the command line with a file path argument
if __name__ == '__main__':
    if len(sys.argv) > 1:
        # The first argument after the script name is the file path
        filepath_from_command = sys.argv[1]
        ingest_document(filepath_from_command)
    else:
        print("Error: Please provide the path to the PDF file as a command-line argument.")
        print("Usage: python ingest.py <path_to_your_file.pdf>")