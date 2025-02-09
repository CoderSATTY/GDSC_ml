from fastapi import FastAPI
from pydantic import BaseModel
import openai
import faiss
import numpy as np
import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# ✅ Load FAISS Index (Replace with your prebuilt FAISS DB)
dimension = 1536  # OpenAI's embedding dimension
index = faiss.IndexFlatL2(dimension)
document_texts = [
    "The sun is a star located at the center of the solar system.",
    "Python is a programming language known for its simplicity.",
    "FastAPI is a high-performance web framework for Python."
]

# Generate Fake Embeddings for Documents (Replace with real embeddings)
document_embeddings = np.random.rand(len(document_texts), dimension).astype("float32")
index.add(document_embeddings)

# ✅ Function to Retrieve Closest Document
def retrieve_relevant_text(query):
    query_embedding = np.random.rand(1, dimension).astype("float32")  # Replace with real embeddings
    _, idxs = index.search(query_embedding, 1)
    return document_texts[idxs[0][0]]

# ✅ Request Model
class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    relevant_text = retrieve_relevant_text(query.text)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": relevant_text}, {"role": "user", "content": query.text}]
    )
    return {"response": response["choices"][0]["message"]["content"]}

# ✅ Speech-to-Text API
@app.post("/voice")
async def voice_chat():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
    text = recognizer.recognize_google(audio)
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": text}]
    )["choices"][0]["message"]["content"]

    # ✅ Text-to-Speech (TTS)
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

    return {"response": response}
