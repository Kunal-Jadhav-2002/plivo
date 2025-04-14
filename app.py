import os
import time
import json
import logging
from psycopg2 import pool
import psycopg2
from flask import Flask, request, Response
from plivo import plivoxml
import openai
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.error(error_msg)
        raise EnvironmentError(error_msg)
    
    logger.info("✅ All environment variables are set")

# Validate environment before proceeding
validate_environment()

# Initialize global variables
openai.api_key = os.getenv("OPENAI_API_KEY")
llm_model_gemini = None
index = None
db_pool = None

def initialize_services():
    """Initialize all external services with error handling"""
    global llm_model_gemini, index, db_pool
    
    try:
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GENAI_API_KEY"))
        llm_model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("✅ Gemini API configured")

        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logger.info("✅ Pinecone client initialized")

        # Setup Pinecone index
        index_name = "voice-bot-gemini-embedding-004-index"
        if index_name not in pc.list_indexes().names():
            logger.info("Creating new Pinecone index...")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("✅ New Pinecone index created")
        else:
            logger.info("✅ Using existing Pinecone index")

        index = pc.Index(index_name)
        logger.info("✅ Connected to Pinecone index")

        # Initialize database connection
        db_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=os.getenv('DATABASE_URL'),
            sslmode='require'
        )
        logger.info("✅ Database connection pool created")

    except Exception as e:
        logger.error(f"❌ Error during service initialization: {str(e)}")
        raise

# Initialize services
initialize_services()

# Initialize other variables
transcript_memory = {}
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")

def request_llm_to_get_summarize(query, context):
    """Generate response using Gemini with error handling"""
    try:
        if not llm_model_gemini:
            raise ValueError("Gemini model not initialized")
            
        logger.info(f"🤖 Processing query: {query[:100]}...")
        user_question_content = f"""
You are a helpful assistant that answers questions about products based on the provided information.

Question: {query}

Context: {context}

Please provide a clear and concise answer based on the context.
"""
        response = llm_model_gemini.generate_content(user_question_content)
        return response.text
    except Exception as e:
        logger.error(f"❌ Error in request_llm_to_get_summarize: {str(e)}")
        return "I apologize, but I'm having trouble generating a response. Please try again."

def generate_text_answer(query_text):
    """Generate answer using vector search with error handling"""
    try:
        if not index:
            raise ValueError("Pinecone index not initialized")
            
        logger.info("🔍 Generating embeddings...")
        query_response = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text
        )
        query_embedding = query_response["embedding"]

        logger.info("🔍 Searching in Pinecone...")
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
        logger.error(f"❌ Error in generate_text_answer: {str(e)}")
        return "I apologize, but I'm having trouble searching for information. Please try again."

def get_ai_response(query):
    """Main function to get AI response with error handling"""
    try:
        if not query or not isinstance(query, str):
            return "Please provide a valid query."
        
        logger.info(f"📝 Processing query: {query}")
        answer = generate_text_answer(query)
        return answer
    except Exception as e:
        logger.error(f"❌ Error in get_ai_response: {str(e)}")
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
    
    # Get content type and log complete request
    content_type = request.headers.get('Content-Type', '')
    print(f"📄 Content-Type: {content_type}")
    print(f"📄 Headers: {dict(request.headers)}")
    
    recording_id = None
    transcription_text = None
    
    # Try JSON format
    if 'application/json' in content_type:
        try:
            data = request.get_json(silent=True)
            if data:
                print(f"🔍 JSON Data: {data}")
                # Try multiple possible field names
                recording_id = data.get("recording_id") or data.get("RecordingID")
                transcription_text = data.get("transcription") or data.get("TranscriptionText")
        except Exception as e:
            print(f"❌ Error processing JSON: {e}")
    
    # Try form data if JSON didn't yield results
    if not recording_id or not transcription_text:
        form_data = request.form.to_dict()
        if form_data:
            print(f"🔍 Form Data: {form_data}")
            recording_id = recording_id or form_data.get("RecordingID") or form_data.get("recording_id")
            transcription_text = transcription_text or form_data.get("TranscriptionText") or form_data.get("transcription")
    
    # Try raw data as last resort
    if not recording_id or not transcription_text:
        try:
            raw_data = request.get_data().decode('utf-8')
            print(f"🔍 Raw Data: {raw_data}")
            if raw_data.startswith('{') and raw_data.endswith('}'):
                try:
                    json_data = json.loads(raw_data)
                    recording_id = recording_id or json_data.get("recording_id") or json_data.get("RecordingID")
                    transcription_text = transcription_text or json_data.get("transcription") or json_data.get("TranscriptionText")
                except:
                    pass
        except Exception as e:
            print(f"❌ Error processing raw data: {e}")
    
    print(f"🎯 Recording ID extracted: {recording_id}")
    print(f"📝 Transcription Text extracted: {transcription_text}")
    
    if recording_id and transcription_text:
        # Convert recording_id to string to ensure consistent keys
        recording_id_str = str(recording_id)
        transcript_memory[recording_id_str] = transcription_text.strip()
        print(f"✅ Saved transcription for Recording ID: {recording_id_str}")
        print(f"✅ Current transcript memory: {transcript_memory}")
        return "OK", 200
    else:
        print("❌ Missing Recording ID or Transcription Text in all attempts")
        return "Missing required fields", 400

@app.route("/process-recording", methods=["POST"])
def process_recording():
    recording_url = request.form.get("RecordUrl")
    recording_id = request.form.get("RecordingID")

    print(f"🎙️ Recording URL: {recording_url}")
    print(f"🆔 Recording ID: {recording_id}")

    # Convert recording_id to string for consistent lookup
    recording_id_str = str(recording_id) if recording_id else None
    
    # Wait for the transcription to be ready with better logging
    transcript = None
    max_attempts = 12  # Increased from 10 to 12
    wait_time = 1.5  # Slightly increased wait time between attempts
    
    print(f"⏳ Waiting for transcription for recording ID: {recording_id_str}")
    print(f"⏳ Current transcript memory keys: {list(transcript_memory.keys())}")
    
    for attempt in range(max_attempts):
        # Check if our recording_id is in the memory dictionary
        if recording_id_str and recording_id_str in transcript_memory:
            transcript = transcript_memory[recording_id_str]
            print(f"✅ Transcription found on attempt {attempt + 1}: {transcript}")
            break
        
        # Try alternate formats of the ID as fallback
        if recording_id:
            # Try with integer format
            try:
                int_id = int(recording_id)
                if str(int_id) in transcript_memory:
                    transcript = transcript_memory[str(int_id)]
                    print(f"✅ Transcription found with integer ID on attempt {attempt + 1}")
                    break
            except:
                pass
        
        # Log the full state of transcript_memory for debugging
        if attempt % 3 == 0:  # Log every 3 attempts to avoid too much output
            print(f"⏳ Transcript memory contents: {transcript_memory}")
        
        print(f"⏳ Attempt {attempt + 1}/{max_attempts}: No transcription yet")
        time.sleep(wait_time)

    if not transcript:
        print("❌ No transcription received after all attempts")
        # Use a more general fallback message
        transcript = "Sorry, I couldn't understand your question. Please try again."

    print(f"📜 Final transcript used: {transcript}")
    reply = get_ai_response(transcript)
    print(f"🤖 AI Response: {reply}")

    response = plivoxml.ResponseElement()
    response.add(plivoxml.SpeakElement(reply))

    # Ask for another query from the user, loop the process
    response.add(plivoxml.SpeakElement("Do you have another question? Please speak after the beep."))
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