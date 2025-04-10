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

# Step 1: Initial call entry point
@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    print("📞 Incoming call:", request.form)

    response = plivoxml.ResponseElement()
    getdigits = plivoxml.GetDigitsElement(
        action="https://web-production-7351.up.railway.app/handle-menu",
        method="POST",
        timeout=10,
        num_digits=1,
        retries=1
    )
    getdigits.add_speak("Press 1 to connect to our assistant. Press 2 to end the call.")
    response.add(getdigits)
    response.add_speak("No input received. Goodbye!")
    return Response(response.to_string(), mimetype='text/xml')

# Step 2: Handle menu selection
@app.route("/handle-menu", methods=["POST"])
def handle_menu():
    digits = request.form.get("Digits")
    print(f"📲 Menu Selection: {digits}")

    response = plivoxml.ResponseElement()

    if digits == "1":
        response.add(
            plivoxml.SpeakElement("Connecting you to the assistant. Please speak after the beep.")
        )
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
    else:
        response.add(plivoxml.SpeakElement("Goodbye!"))

    return Response(response.to_string(), mimetype='text/xml')

# Step 3: Receive transcription from Plivo
@app.route("/transcription", methods=["POST"])
def save_transcription():
    recording_url = request.form.get("RecordUrl")
    transcription_text = request.form.get("TranscriptionText")

    print(f"📝 Transcription Received:\nURL: {recording_url}\nText: {transcription_text}")

    if recording_url and transcription_text:
        transcript_memory[recording_url] = transcription_text.strip()
    return "OK", 200

# Step 4: Process recording after transcription
@app.route("/process-recording", methods=["POST"])
def process_recording():
    recording_url = request.form.get("RecordUrl")
    print(f"🎙️ Recording URL: {recording_url}")

    # Wait until transcription is ready
    transcript = None
    for _ in range(10):
        transcript = transcript_memory.get(recording_url)
        if transcript:
            break
        time.sleep(1)

    if not transcript:
        transcript = "Sorry, I couldn't understand. Could you please repeat?"

    print(f"📜 Transcript used: {transcript}")

    # Get AI reply
    reply = get_ai_response(transcript)

    # Respond with AI reply and repeat
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

# Step 5: Get response from OpenAI
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
        return "Sorry, something went wrong with our assistant."

# Run Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)