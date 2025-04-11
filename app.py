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

# Load environment variables
load_dotenv()
app = Flask(__name__)

def validate_environment():
    """Validate all required environment variables and API keys"""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for GPT-4",
        "PLIVO_AUTH_ID": "Plivo authentication ID",
        "PLIVO_AUTH_TOKEN": "Plivo authentication token",
        "DATABASE_URL": "PostgreSQL database URL",
        "GENAI_API_KEY": "Google Gemini API key",
        "PINECONE_API_KEY": "Pinecone API key"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        error_msg = "Missing required environment variables:\n" + "\n".join(f"- {var}" for var in missing_vars)
        print(f"❌ {error_msg}")
        raise EnvironmentError(error_msg)
    
    print("✅ All environment variables are set")

# Validate environment before proceeding
validate_environment()

# Initialize APIs with error handling
try:
    # Initialize OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI client initialized")

    # Initialize Gemini
    genai.configure(api_key=os.getenv("GENAI_API_KEY"))
    llm_model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Gemini API configured")

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    print("✅ Pinecone client initialized")

    # Setup Pinecone index
    index_name = "voice-bot-gemini-embedding-004-index"
    if index_name not in pc.list_indexes().names():
        print("Creating new Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("✅ New Pinecone index created")
    else:
        print("✅ Using existing Pinecone index")

    index = pc.Index(index_name)
    print("✅ Connected to Pinecone index")

except Exception as e:
    print(f"❌ Error during API initialization: {str(e)}")
    raise

# Initialize database connection
try:
    db_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=5,
        dsn=os.getenv('DATABASE_URL'),
        sslmode='require'
    )
    print("✅ Database connection pool created")
except Exception as e:
    print(f"❌ Database connection error: {str(e)}")
    raise

# Initialize other variables
transcript_memory = {}
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")

def request_llm_to_get_summarize(query, context):
    """Generate response using Gemini with error handling"""
    try:
        print(f"🤖 Processing query: {query[:100]}...")
        user_question_content = f"""
You are a highly accurate and detail-oriented question-answering assistant. Your task is to help users by answering their questions about products based on the provided search results.

### Current User Question:
{query}

### Search Results:
{context}

Please provide a clear and concise answer based on the search results.
"""
        response = llm_model_gemini.generate_content(user_question_content)
        return response.text
    except Exception as e:
        print(f"❌ Error in request_llm_to_get_summarize: {str(e)}")
        return "I apologize, but I'm having trouble generating a response. Please try again."

def generate_text_answer(query_text):
    """Generate answer using vector search with error handling"""
    try:
        print("🔍 Generating embeddings...")
        query_response = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text
        )
        query_embedding = query_response["embedding"]

        print("🔍 Searching in Pinecone...")
        search_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        context_lst = []
        for match in search_results["matches"]:
            if "metadata" in match and "text" in match["metadata"]:
                context_lst.append(match["metadata"]["text"])

        if not context_lst:
            return "I couldn't find any relevant information about that product."

        context = "\n-----------------\n".join(context_lst)
        return request_llm_to_get_summarize(query_text, context)

    except Exception as e:
        print(f"❌ Error in generate_text_answer: {str(e)}")
        return "I apologize, but I'm having trouble searching for information. Please try again."

def get_ai_response(query):
    """Main function to get AI response with error handling"""
    try:
        if not query or not isinstance(query, str):
            return "Please provide a valid query."
        
        print(f"📝 Processing query: {query}")
        answer = generate_text_answer(query)
        return answer
    except Exception as e:
        print(f"❌ Error in get_ai_response: {str(e)}")
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