import os
import time
import json
from psycopg2 import pool
import psycopg2
from flask import Flask, request, Response
from plivo import plivoxml
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
app = Flask(__name__)

# Set up API keys
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)
llm_model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define Pinecone index name
index_name = "voice-bot-gemini-embedding-004-index"

# Check if index exists, otherwise create it
if index_name not in pc.list_indexes().names():
    print("Creating a new index!")
    pc.create_index(
        name=index_name,
        dimension=768,  # GEMINI 'text-embedding-004' returns 768-dim vectors
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print("Index already exists, connecting to it!")

# Connect to the index
index = pc.Index(index_name)
print("Index loaded and index: ", index)

# Check for required environment variables
required_env_vars = [
    "OPENAI_API_KEY",
    "PLIVO_AUTH_ID",
    "PLIVO_AUTH_TOKEN",
    "DATABASE_URL",
    "GENAI_API_KEY",
    "PINECONE_API_KEY"
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

def request_llm_to_get_summarize(query, context):
    user_question_content = f"""
You are a highly accurate and detail-oriented question-answering assistant. Your task is to help users by answering their questions about products based on the provided search results. The search results contain information about multiple products, and each product has the following details:

- **PRODUCT NAME**: The name of the product.
- **Product Quantity**: The available quantity of the product.
- **Product Rate**: The price of a single unit of the product in INR.
- **Product Value**: The total value of the product, calculated as (Product Quantity × Product Rate).

### Instructions:
1. Carefully analyze the user's question and the provided search results.
2. Answer the user's question **only** using the information from the search results. Do not make up or assume any details.
3. User queries initially come in as voice recordings and are converted to text using a speech-to-text model. However, speech-to-text models might not always accurately capture product names mentioned in the query. When the exact product from the user's question is not identified in the search results:
   - Suggest any closely matching products (if available) and provide their details.
   - If no closely matching products available in search results, say that the requested product information is not available.
4. Keep your response concise, accurate, and directly relevant to the user's question.

### Current User Question:
{query}

### Search Results:
{context}

### Your Task:
Provide a clear and concise answer to the user's question based on the search results.
"""
    response = llm_model_gemini.generate_content(user_question_content)
    return response.text

def generate_text_answer(query_text):
    print("query_text input to generate_text_answer: ", query_text)
    query_response = genai.embed_content(model="models/text-embedding-004", content=query_text)
    query_embedding = query_response["embedding"]

    # Search in Pinecone
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # Extract search results
    context_lst = []
    for match in search_results["matches"]:
        chunk_text = match["metadata"]["text"]
        context_lst.append(chunk_text)

    context = "\n-----------------\n".join(context_lst)
    answer = request_llm_to_get_summarize(query_text, context)
    return answer

def get_ai_response(query):
    try:
        answer = generate_text_answer(query)
        return answer
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "I apologize, but I'm having trouble understanding that. Could you please repeat your question?"

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
        transcript = "Sorry, I couldn't understand. Could you please repeat?"

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