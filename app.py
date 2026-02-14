from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import requests

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ------------------------
# Create FastAPI app
# ------------------------

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Global Variables
# ------------------------

vectorstore = None
retriever = None
PERSIST_DIRECTORY = "./chroma_db"

# ------------------------
# Startup Event
# ------------------------

@app.on_event("startup")
def startup_event():
    global vectorstore, retriever

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(PERSIST_DIRECTORY):
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    else:
        loader = TextLoader("data/data.txt")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )

        vectorstore.persist()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ------------------------
# Request Model
# ------------------------

class ChatRequest(BaseModel):
    message: str

# ------------------------
# Routes
# ------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: ChatRequest):

    docs = await asyncio.to_thread(
        retriever.invoke,
        request.message
    )

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {request.message}
    """

    API_KEY = os.getenv("GEMINI_API_KEY")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = await asyncio.to_thread(requests.post, url, json=payload)
    data = response.json()

    answer = data["candidates"][0]["content"]["parts"][0]["text"]

    return {"response": answer}