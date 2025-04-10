import os
from flask import Flask, request, Response
from plivo import plivoxml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask is up and running!"

# Menu when call arrives
@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    response = plivoxml.ResponseElement()
    get_digits = plivoxml.GetDigitsElement(
        action="https://web-production-7351.up.railway.app/handle-menu",
        method="POST",
        timeout=5,
        num_digits=1,
        retries=1
    )
    get_digits.add_speak("Welcome to Tecnvirons. Press 1 to talk to our assistant. Press 2 to end the call.")
    response.add(get_digits)
    response.add_speak("We didn’t get any input. Goodbye!")
    return Response(response.to_string(), mimetype="text/xml")

# Handle menu choices
@app.route("/handle-menu", methods=["POST"])
def handle_menu():
    digit = request.form.get("Digits")
    response = plivoxml.ResponseElement()

    if digit == "1":
        response.add_speak("You will now be connected to our assistant.")
        response.add(
            plivoxml.RecordElement(
                action="https://web-production-7351.up.railway.app/recording-complete",
                method="POST",
                max_length=30,
                timeout=5,
                transcription_type="auto",
                transcription_url="https://web-production-7351.up.railway.app/transcription",
                transcription_method="POST",
                play_beep=True
            )
        )
    elif digit == "2":
        response.add_speak("Thank you for calling. Goodbye.")
    else:
        response.add_speak("Invalid input. Ending call.")

    return Response(response.to_string(), mimetype="text/xml")

# (Optional) This endpoint is required but not used
@app.route("/recording-complete", methods=["POST"])
def recording_complete():
    print("🎤 Recording complete. Awaiting transcription...")
    return "OK", 200

# When transcription is ready, this handles the logic!
@app.route("/transcription", methods=["POST"])
def transcription():
    transcription_text = request.form.get("TranscriptionText")
    recording_id = request.form.get("RecordingID")

    print(f"📝 Transcription Received:\nRecording ID: {recording_id}\nText: {transcription_text}")

    if not transcription_text:
        transcription_text = "Sorry, I didn't catch that. Could you please repeat?"

    reply = get_ai_response(transcription_text)

    # Respond back with reply and re-record
    response = plivoxml.ResponseElement()
    response.add(plivoxml.SpeakElement(reply))
    response.add(
        plivoxml.RecordElement(
            action="https://web-production-7351.up.railway.app/recording-complete",
            method="POST",
            max_length=30,
            timeout=5,
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
        return "Something went wrong. Please try again later."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)