from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from google import genai
import os
from google.genai import types

load_dotenv()

# 1. Setup Embeddings (Must be the SAME model as used in Indexing)
# If you use a different model, the "language" of the numbers won't match.
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# 2. Connect to the Existing Vector Store
# We don't use 'from_documents' because we aren't adding new data.
# We use 'from_existing_collection' to connect to what we already saved.
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="learning_rag",
    url="http://localhost:6333"
)

def chat_loop():
    # Initialize the client once
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # 1. Initialize Chat History (The "Transcript")
    # We start with an empty list to store the conversation
    chat_history = []
    
    print("---------------------------------------------------------")
    print("🤖 RAG Chatbot Ready! (Type 'exit', 'quit' or 'q' to stop)")
    print("---------------------------------------------------------")

    while True:
        # Take user query input inside the loop
        user_query = input("\n👤 User: ")
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye! 👋")
            break
            
        if not user_query.strip():
            continue

        print("🔍 Searching documents...")
        
        # Relevant chunks from the vector store
        search_results = vector_store.similarity_search(query=user_query)

        context = "\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}\nFile Location: {result.metadata.get('source', 'Unknown')}" 
                                for result in search_results])

        # 2. Construct the Prompt with History
        # We combine the System Prompt + Chat History + Current Question
        
        # Format history for the prompt
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        
        full_prompt = f"""You are a helpful AI assistant.
        
CONTEXT FROM DOCUMENTS:
{context}

CHAT HISTORY:
{history_text}

CURRENT USER QUESTION:
{user_query}

INSTRUCTIONS:
1. First, check the "CONTEXT FROM DOCUMENTS". If the answer is there, use it.
1.1 If the answer is not clear, ask clarifying questions based on the context.
1.2 Also add a reference to the source (file name and page number) in separate line every time you use the context.
2. If the answer is NOT in the documents, check the "CHAT HISTORY" to see if this is a follow-up.
3. If you still cannot answer, you are allowed to use Google Search to find the answer.
4. If you use Google Search, explicitly mention that you found the information online.
"""

        try:
            # Make the API call
            # We enable the 'google_search' tool here
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(
                        google_search=types.GoogleSearch() # Giving it the "Search Tool"
                    )]
                )
            )

            # Print the response
            ai_response = response.text
            print(f"🤖 Gemini: {ai_response}")
            
            # 3. Update History
            # Add the interaction to our transcript so we remember it next time
            chat_history.append({"role": "User", "content": user_query})
            chat_history.append({"role": "AI", "content": ai_response})
            
            # Keep history short (optional, to save tokens)
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    chat_loop()

