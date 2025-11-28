from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# LangChain & Gemini Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from google.genai import types

# Load env vars
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "learning_rag"

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# --- Data Models ---
class ChatRequest(BaseModel):
    message: str
    history: List[dict] = [] # List of {"role": "user"|"model", "content": "..."}

class ChatResponse(BaseModel):
    response: str

# --- Routes ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 1. Save the file locally
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Process the PDF (Indexing Logic)
        loader = PyPDFLoader(file_path=str(file_path))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        chunks = text_splitter.split_documents(documents=docs)
        
        # 3. Store in Qdrant
        # Note: force_recreate=True to overwrite old data for this demo, 
        # or remove it to append.
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            force_recreate=True 
        )
        
        return {"message": f"Successfully processed {file.filename}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 1. Connect to Vector Store
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embedding_model,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL
        )
        
        # 2. Retrieve Context
        search_results = vector_store.similarity_search(query=request.message, k=3)
        context_text = "\n\n".join([
            f"Content: {res.page_content}\nSource: {res.metadata.get('source', 'Unknown')}" 
            for res in search_results
        ])
        
        # 3. Format History for Gemini
        # Gemini expects specific roles or we can just format it as text in the prompt
        # For simplicity and robustness with the 'generate_content' API, we'll format it into the prompt
        # or use the contents list if we want to maintain state properly.
        
        # Let's construct a rich prompt
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.history])
        
        system_prompt = f"""You are a helpful AI assistant.
Use the provided context to answer the user's question.
If the answer is not in the context, you can use your internal knowledge.
Always prioritize the provided context.

CONTEXT FROM UPLOADED DOCUMENTS:
{context_text}

CHAT HISTORY:
{history_text}

CURRENT USER QUESTION:
{request.message}
"""
        
        # 4. Call Gemini
        # We can enable Google Search if we want, similar to the CLI version
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=system_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3
            )
        )
        
        return {"response": response.text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
