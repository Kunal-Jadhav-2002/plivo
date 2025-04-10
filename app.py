import os
from flask import Flask, request, Response
from plivo import plivoxml
import openai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

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
            action="/process-recording",
            method="POST",
            max_length=30,
            timeout=10,
            transcription_type="auto",
            transcription_url="/transcription",
            transcription_method="POST",
            play_beep=True
        )
    )
    return Response(response.to_string(), mimetype='text/xml')

@app.route("/process-recording", methods=["POST"])
def process_recording():
    recording_url = request.form.get("RecordUrl")
    print(f"🎙️ Recording URL: {recording_url}")

    # Placeholder for transcription step (use Whisper or manual transcription here)
    transcript = "What are your working hours?"  # TODO: Use Whisper or similar service

    # Get response from OpenAI
    reply = get_ai_response(transcript)

    # Create next XML response
    response = plivoxml.ResponseElement()
    response.add(plivoxml.SpeakElement(reply))
    response.add(
        plivoxml.RecordElement(
            action="/process-recording",
            method="POST",
            max_length=30,
            timeout=10,
            transcription_type="auto",
            transcription_url="/transcription",
            transcription_method="POST",
            play_beep=True
        )
    )
    return Response(response.to_string(), mimetype="text/xml")

def get_ai_response(query):
    try:
        completion = openai.ChatCompletion.create(
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