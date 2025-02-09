import streamlit as st
import requests
import speech_recognition as sr
import pyttsx3

# Backend API URL
BACKEND_URL = "http://127.0.0.1:8000"

st.title("🔹 RAG + Voice Chatbot")

# ✅ User Input
query = st.text_input("Ask a question:")

if st.button("Send"):
    if query:
        response = requests.post(f"{BACKEND_URL}/chat", json={"text": query})
        st.write("🤖 Chatbot:", response.json()["response"])

# ✅ Voice Interaction
if st.button("🎤 Speak"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        st.write(f"🎙 You said: {text}")
        
        response = requests.post(f"{BACKEND_URL}/chat", json={"text": text})
        st.write("🤖 Chatbot:", response.json()["response"])

        # ✅ Text-to-Speech Response
        engine = pyttsx3.init()
        engine.say(response.json()["response"])
        engine.runAndWait()

    except Exception as e:
        st.error("Error recognizing speech. Please try again.")
