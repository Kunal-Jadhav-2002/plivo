import os
import time
import json
from psycopg2 import pool
import psycopg2
from flask import Flask, request, Response
from plivo import plivoxml
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Check for required environment variables
required_env_vars = [
    "OPENAI_API_KEY",
    "PLIVO_AUTH_ID",
    "PLIVO_AUTH_TOKEN",
    "DATABASE_URL"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
transcript_memory = {}

# Plivo Authentication
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")

# Database connection pool
try:
    db_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=os.getenv('DATABASE_URL'),
        sslmode='require'
    )
except Exception as e:
    print(f"❌ Database connection error: {e}")
    raise

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
        # First, use GPT-4 to extract the product name from the query
        product_extraction = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a product name extractor. Your task is to identify the product name from the user's query.
                    Return ONLY the product name, nothing else. If no clear product name is found, return 'NO_PRODUCT'."""
                },
                {"role": "user", "content": query}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        product_name = product_extraction.choices[0].message.content.strip()
        print(f"🔍 Extracted product name: {product_name}")
        
        if product_name != "NO_PRODUCT":
            # Try to get product details from database
            conn = db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    # Search for products with similar names
                    cur.execute("""
                        SELECT product_name, product_quantity, product_rate, product_value, description
                        FROM products
                        WHERE LOWER(product_name) LIKE LOWER(%s)
                        LIMIT 1;
                    """, (f"%{product_name}%",))
                    result = cur.fetchone()
                    
                    if result:
                        product_details = {
                            "product_name": result[0],
                            "quantity": result[1],
                            "rate": result[2],
                            "value": result[3],
                            "description": result[4]
                        }
                        
                        # Create a detailed response using the product information
                        response = f"""The {product_details['product_name']} is available in our inventory.
                        Available Quantity: {product_details['quantity']}
                        Price per Unit: {product_details['rate']}
                        Total Value: {product_details['value']}
                        Description: {product_details['description']}"""
                        
                        return response
            finally:
                db_pool.putconn(conn)
        
        # If no product found or no product name extracted, use GPT-4 for a general response
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
        action="https://web-production-aa492.up.railway.app/handle-menu",
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
                action="https://web-production-aa492.up.railway.app/process-recording",
                method="POST",
                max_length=30,
                timeout=5,
                transcription_type="auto",
                transcription_url="https://web-production-aa492.up.railway.app/transcription",
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
    
    # Get content type
    content_type = request.headers.get('Content-Type', '')
    print(f"📄 Content-Type: {content_type}")
    
    # Handle JSON data
    if 'application/json' in content_type:
        try:
            data = request.get_json()
            print("🔍 JSON Data:", data)
            
            recording_id = data.get("recording_id")
            transcription_text = data.get("transcription")
            
            print(f"🎯 Recording ID from JSON: {recording_id}")
            print(f"📝 Transcription Text from JSON: {transcription_text}")
            
            if recording_id and transcription_text:
                transcript_memory[recording_id] = transcription_text.strip()
                print(f"✅ Saved transcription for Recording ID: {recording_id}")
                return "OK", 200
            else:
                print("❌ Missing Recording ID or Transcription Text in JSON")
                return "Missing required fields", 400
                
        except Exception as e:
            print(f"❌ Error processing JSON: {e}")
            return str(e), 400
    
    # Handle form data as fallback
    recording_id = request.form.get("RecordingID")
    transcription_text = request.form.get("TranscriptionText")
    
    print(f"🎯 Recording ID from form: {recording_id}")
    print(f"📝 Transcription Text from form: {transcription_text}")
    
    if recording_id and transcription_text:
        transcript_memory[recording_id] = transcription_text.strip()
        print(f"✅ Saved transcription for Recording ID: {recording_id}")
        return "OK", 200
    else:
        print("❌ Missing Recording ID or Transcription Text in form data")
        return "Missing required fields", 400

@app.route("/process-recording", methods=["POST"])
def process_recording():
    recording_url = request.form.get("RecordUrl")
    recording_id = request.form.get("RecordingID")

    print(f"🎙️ Recording URL: {recording_url}")
    print(f"🆔 Recording ID: {recording_id}")

    # Wait for the transcription to be ready with better logging
    transcript = None
    max_attempts = 10  # Increased from 8 to 10
    wait_time = 1  # seconds between attempts
    
    print("⏳ Waiting for transcription...")
    for attempt in range(max_attempts):
        transcript = transcript_memory.get(recording_id)
        if transcript:
            print(f"✅ Transcription received on attempt {attempt + 1}: {transcript}")
            break
        print(f"⏳ Attempt {attempt + 1}/{max_attempts}: No transcription yet")
        time.sleep(wait_time)

    if not transcript:
        print("❌ No transcription received after all attempts")
        transcript = "Sorry, I couldn't understand. Could you please repeat ? my phone number is 7058032981 can you tell me about Push Button 020 Red from database."

    print(f"📜 Final transcript used: {transcript}")
    reply = get_ai_response(transcript)
    print(f"🤖 AI Response: {reply}")

    response = plivoxml.ResponseElement()
    response.add(plivoxml.SpeakElement(reply))

    # Ask for another query from the user, loop the process
    response.add(plivoxml.RecordElement(
        action="https://web-production-aa492.up.railway.app/process-recording",
        method="POST",
        max_length=30,
        timeout=5,
        transcription_type="auto",
        transcription_url="https://web-production-aa492.up.railway.app/transcription",
        transcription_method="POST",
        play_beep=True
    ))

    return Response(response.to_string(), mimetype="text/xml")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)