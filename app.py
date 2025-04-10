import os
import time
from flask import Flask, request, Response
from plivo import plivoxml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory store for storing transcriptions
transcript_memory = {}

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask server is running!"

@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    print("📞 Incoming call from Plivo:")
    print(request.form)

    response = plivoxml.ResponseElement()
    response.add(
        plivoxml.SpeakElement("Hello, welcome to Tecnvirons. How may I assist you?")
    )
    response.add(
        plivoxml.RecordElement(
            action="",
            method="POST",
            max_length=30,
            timeout=10,
            transcription_type="auto",
            transcription_url="https://web-production-7351.up.railway.app/transcription",
            transcription_method="POST",
            play_beep=True
        )
    )
    return Response(response.to_string(), mimetype='text/xml')

@app.route("/transcription", methods=["POST"])
def transcription_handler():
    recording_id = request.form.get("RecordingID")
    transcription_text = request.form.get("TranscriptionText")

    print(f"📝 Transcription Received:\nRecording ID: {recording_id}\nText: {transcription_text}")

    if not transcription_text:
        transcription_text = "Sorry, I couldn't understand. Could you please repeat?"

    # Generate AI response
    reply = get_ai_response(transcription_text.strip())

    # Respond with AI-generated message and ask again
    response = plivoxml.ResponseElement()
    response.add(plivoxml.SpeakElement(reply))
    response.add(
        plivoxml.RecordElement(
            action="",  # Still not needed
            method="POST",
            max_length=30,
            timeout=10,
            transcription_type="auto",
            transcription_url="https://web-production-7351.up.railway.app/transcription",
            transcription_method="POST",
            play_beep=True
        )
    )
    return Response(response.to_string(), mimetype="text/xml")

def get_ai_response(query):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return "Sorry, I couldn't understand that. Please repeat."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)