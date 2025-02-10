from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import groq
import faiss
import numpy as np
import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv

# ✅ Load API Key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is missing! Check your .env file.")

# ✅ Initialize Groq client
groq_client = groq.Groq(api_key=GROQ_API_KEY)

app = FastAPI()

# ✅ FAISS Setup
dimension = 1536
index = faiss.IndexFlatL2(dimension)
document_texts = [
    "The sun is a star at the center of the solar system.",
    "Python is a programming language known for its simplicity.",
    "FastAPI is a high-performance web framework for Python."
]
document_embeddings = np.random.rand(len(document_texts), dimension).astype("float32")
index.add(document_embeddings)

def retrieve_relevant_text(query):
    query_embedding = np.random.rand(1, dimension).astype("float32")
    _, idxs = index.search(query_embedding, 1)
    return document_texts[idxs[0][0]]

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        relevant_text = retrieve_relevant_text(query.text)
        response = groq_client.chat.completions.create(
            model="mixtral-8x7b",  # ✅ Groq's free & fast model
            messages=[
                {"role": "system", "content": relevant_text},
                {"role": "user", "content": query.text}
            ]
        )
        print("🔍 Groq Response:", response)  # ✅ DEBUG
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": f"❌ Error processing request: {str(e)}"}

# ✅ Speech-to-Text API
@app.post("/voice")
async def voice_chat():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        
        response = groq_client.chat.completions.create(
            model="llama3-8b",
            messages=[{"role": "user", "content": text}]
        )

        response_text = response.choices[0].message.content

        # ✅ Text-to-Speech (TTS)
        engine = pyttsx3.init()
        engine.say(response_text)
        engine.runAndWait()

        return {"response": response_text}
    except Exception as e:
        return {"error": f"❌ Error processing voice request: {str(e)}"}
