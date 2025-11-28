# RAG-Based PDF Chatbot

A full-stack application that allows users to upload PDF documents and chat with them using RAG (Retrieval-Augmented Generation). The application uses Google's Gemini Pro model for generation and Qdrant for vector storage.

## Features

- 📄 **PDF Upload**: Upload and index PDF documents.
- 💬 **Interactive Chat**: Chat with your documents using natural language.
- 🧠 **RAG Pipeline**: Uses LangChain and Google Generative AI embeddings.
- 🎨 **Modern UI**: Built with React and Vite, featuring dark mode and Markdown support.
- ⚡ **Fast Backend**: Powered by FastAPI.

## Tech Stack

- **Frontend**: React, Vite, Axios, React Markdown
- **Backend**: FastAPI, Uvicorn, Python
- **AI/ML**: Google Gemini (via `google-genai`), LangChain
- **Vector DB**: Qdrant (Docker)

## Prerequisites

- Python 3.10+
- Node.js & npm
- Docker Desktop (for Qdrant)
- Google API Key

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/parthpatidar03/RAG-Based-Pdf-Chatbot.git
cd RAG-Based-Pdf-Chatbot
```

### 2. Start Qdrant Vector DB
Ensure Docker is running, then start the Qdrant container:
```bash
docker-compose up -d
```

### 3. Backend Setup
Navigate to the root directory and install dependencies:
```bash
pip install -r requirements.txt
```
Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```
Start the backend server:
```bash
python -m uvicorn server:app --reload
```
The backend will run at `http://localhost:8000`.

### 4. Frontend Setup
Navigate to the frontend directory:
```bash
cd frontend
npm install
```
Start the development server:
```bash
npm run dev
```
The frontend will run at `http://localhost:5173`.

## Usage

1. Open the frontend URL.
2. Upload a PDF file using the "Upload PDF" button.
3. Once processed, type your questions in the chat box and send!
