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
    get_digits = plivoxml.GetDigitsElement(
        action="https://web-production-7351.up.railway.app/handle-menu",
        method="POST",
        timeout=10,
        num_digits=1,
        retries=1
    )
    get_digits.add_speak("Press 1 to talk to our virtual assistant. Press 2 to end the call.")
    response.add(get_digits)
    response.add_speak("No input received. Goodbye!")
    return Response(response.to_string(), mimetype='text/xml')

@app.route("/handle-menu", methods=["POST"])
def handle_menu():
    digit = request.form.get("Digits")
    print(f"📲 Menu Selection: {digit}")
    
    response = plivoxml.ResponseElement()

    if digit == "1":
        response.add(plivoxml.SpeakElement("Connecting you to our assistant."))
        response.add(
            plivoxml.RecordElement(
                action="https://web-production-7351.up.railway.app/process-recording",
                method="POST",
                max_length=30,
                timeout=10,
                transcription_type="auto",
                transcription_url="https://web-production-7351.up.railway.app/transcription",
                transcription_method="POST",
                play_beep=True
            )
        )
    elif digit == "2":
        response.add(plivoxml.SpeakElement("Thank you for calling. Goodbye!"))
    else:
        response.add(plivoxml.SpeakElement("Invalid input. Goodbye!"))

    return Response(response.to_string(), mimetype='text/xml')

@app.route("/transcription", methods=["POST"])
def save_transcription():
    recording_id = request.form.get("RecordingID")
    transcription_text = request.form.get("TranscriptionText")
    print(f"📝 Transcription Received:\nRecording ID: {recording_id}\nText: {transcription_text}")

    if recording_id and transcription_text:
        transcript_memory[recording_id] = transcription_text.strip()
    return "OK", 200

@app.route("/process-recording", methods=["POST"])
def process_recording():
    recording_url = request.form.get("RecordUrl")
    recording_id = request.form.get("RecordingID")

    print(f"🎙️ Recording URL: {recording_url}")
    print(f"🆔 Recording ID: {recording_id}")

    transcript = None
    for _ in range(10):
        transcript = transcript_memory.get(recording_id)
        if transcript:
            break
        time.sleep(1)

    if not transcript:
        transcript = "Sorry, I couldn't understand. Could you please repeat?"

    print(f"📜 Transcript used: {transcript}")

    reply = get_ai_response(transcript)

    response = plivoxml.ResponseElement()
    response.add(plivoxml.SpeakElement(reply))
    response.add(
        plivoxml.RecordElement(
            action="https://web-production-7351.up.railway.app/process-recording",
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