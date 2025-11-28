from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader # extaract text from PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # for converting text into vector embeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

# Define the path to the PDF file
# Using the file found in the directory: "Test Pdf Metallurgy.pdf"
pdf_path = Path(__file__).parent / "Test Pdf Metallurgy.pdf"

# Load this file in python program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load() # List of documents loaded from the PDF file 

# Split the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(documents=docs)

# Vector Embeddings
# Ensure GOOGLE_API_KEY is set in your environment variables
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)
# Main Functionality - Indexing the documents into Qdrant Vector Store 
# Create Vector Store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)

print("Indexing of documents done....")