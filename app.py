import os
import time
import json
import psycopg2
from flask import Flask, request, Response
from plivo import plivoxml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
transcript_memory = {}

# Plivo Authentication
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")

# Database connection pool
db_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=5,
    dsn=os.getenv('DATABASE_URL'),
    sslmode='require'
)

def insert_user_data(username, email, phone_number):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (username, email, phone_number)
                VALUES (%s, %s, %s);
            """, (username, email, phone_number))
            conn.commit()
        return True
    finally:
        db_pool.putconn(conn)

def get_user_by_phone(phone_number):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            normalized_phone = ''.join(filter(str.isdigit, str(phone_number)))
            cur.execute("""
                SELECT username, email
                FROM users
                WHERE phone_number = %s
                LIMIT 1;
            """, (normalized_phone,))
            result = cur.fetchone()
            return {'username': result[0], 'email': result[1]} if result else False
    except Exception as e:
        print(f"Database error: {e}")
        return False
    finally:
        db_pool.putconn(conn)

def get_ai_response(query):
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful product information assistant. Your task is to help users by answering their questions about products. 
                    When answering questions:
                    1. Be concise and clear
                    2. If you don't have specific information about a product, say so
                    3. If the user's question is unclear, ask for clarification
                    4. Keep responses brief and to the point
                    5. If the user wants to book a meeting, provide the calendly link: https://calendly.com/ai-tecnvi-ai/30min"""
                },
                {"role": "user", "content": query}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return "Sorry, I couldn't understand that. Please repeat."

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask is running!"

@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    print("📞 Incoming call:", request.form)

    response = plivoxml.ResponseElement()
    get_digits = plivoxml.GetDigitsElement(
        action="https://web-production-675e.up.railway.app/handle-menu",
        method="POST",
        timeout=10,
        num_digits=1,
        retries=1
    )
    get_digits.add_speak("Welcome to Tecnvirons. Press 1 to talk to our AI assistant. Press 2 to end the call.")
    response.add(get_digits)
    response.add_speak("No input received. Goodbye.")
    return Response(response.to_string(), mimetype='text/xml')

@app.route("/handle-menu", methods=["POST"])
def handle_menu():
    digit = request.form.get("Digits")
    print(f"📲 Menu Selection: {digit}")

    response = plivoxml.ResponseElement()
    if digit == "1":
        response.add(plivoxml.SpeakElement("Please describe your query after the beep."))
        response.add(
            plivoxml.RecordElement(
                action="https://web-production-675e.up.railway.app/process-recording",
                method="POST",
                max_length=30,
                timeout=5,
                transcription_type="auto",
                transcription_url="https://web-production-675e.up.railway.app/transcription",
                transcription_method="POST",
                play_beep=True
            )
        )
    else:
        response.add(plivoxml.SpeakElement("Thank you for calling. Goodbye."))

    return Response(response.to_string(), mimetype='text/xml')

@app.route("/transcription", methods=["POST"])
def save_transcription():
    print("📝 Transcription Callback Received")
    print("🔍 Raw form data:", request.form)

    recording_id = request.form.get("RecordingID")
    transcription_text = request.form.get("TranscriptionText")

    print(f"Recording ID: {recording_id}")
    print(f"Text: {transcription_text}")

    if recording_id and transcription_text:
        transcript_memory[recording_id] = transcription_text.strip()

    return "OK", 200

@app.route("/process-recording", methods=["POST"])
def process_recording():
    recording_url = request.form.get("RecordUrl")
    recording_id = request.form.get("RecordingID")

    print(f"🎙️ Recording URL: {recording_url}")
    print(f"🆔 Recording ID: {recording_id}")

    # Wait for the transcription to be ready
    transcript = None
    for _ in range(8):
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

    # Ask for another query from the user, loop the process
    response.add(plivoxml.RecordElement(
        action="https://web-production-675e.up.railway.app/process-recording",
        method="POST",
        max_length=30,
        timeout=5,
        transcription_type="auto",
        transcription_url="https://web-production-675e.up.railway.app/transcription",
        transcription_method="POST",
        play_beep=True
    ))

    return Response(response.to_string(), mimetype="text/xml")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)