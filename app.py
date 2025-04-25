import os
import time
import json
import logging
import asyncio
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import psycopg2
from psycopg2 import pool
from flask import Flask, request, Response, g
from flask_socketio import SocketIO, emit
from plivo import plivoxml
import openai
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=10)

# In-memory cache for responses and transcriptions
response_cache = {}
transcription_cache = {}
active_calls = {}

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

# Initialize globals
openai.api_key = os.getenv("OPENAI_API_KEY")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")

# Service connection objects
llm_model_gemini = None
index = None
db_pool = None

def initialize_services():
    """Initialize all external services with error handling - runs at startup"""
    global llm_model_gemini, index, db_pool
    
    # Initialize services in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        gemini_future = executor.submit(initialize_gemini)
        pinecone_future = executor.submit(initialize_pinecone)
        db_future = executor.submit(initialize_database)
        
        # Wait for all to complete and handle any errors
        futures = [gemini_future, pinecone_future, db_future]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"❌ Service initialization error: {str(e)}")
                raise

def initialize_gemini():
    """Initialize Gemini AI - runs in parallel"""
    global llm_model_gemini
    genai.configure(api_key=os.getenv("GENAI_API_KEY"))
    # Pre-load the model to reduce cold start
    llm_model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    # Warmup query to initialize the model
    _ = llm_model_gemini.generate_content("Hello")
    logger.info("✅ Gemini API configured and warmed up")

def initialize_pinecone():
    """Initialize Pinecone vector database - runs in parallel"""
    global index
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
    
    # Warm up the index with a simple query
    try:
        # Create a zero vector for warm-up
        zero_vector = [0.0] * 768
        index.query(vector=zero_vector, top_k=1, include_metadata=True)
        logger.info("✅ Pinecone index connection verified and warmed up")
    except Exception as e:
        logger.warning(f"⚠️ Pinecone warm-up query failed: {str(e)}")

def initialize_database():
    """Initialize database connection pool - runs in parallel"""
    global db_pool
    db_pool = pool.ThreadedConnectionPool(
        minconn=5,  # Increased from 1 to handle concurrent requests
        maxconn=20, # Increased from 5 to handle peak load
        dsn=os.getenv('DATABASE_URL'),
        sslmode='require'
    )
    
    # Test connection by getting and immediately returning a connection
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        db_pool.putconn(conn)
        logger.info("✅ Database connection pool created and verified")
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {str(e)}")
        raise

# Run initialization
logger.info("🚀 Starting service initialization...")
initialize_services()
logger.info("✅ All services initialized successfully")

# Database connection helpers
def get_db_connection():
    """Get a database connection from the pool with retry logic"""
    max_retries = 3
    retry_delay = 0.5
    
    for attempt in range(max_retries):
        try:
            if not hasattr(g, 'db_conn'):
                g.db_conn = db_pool.getconn() if db_pool else None
                if g.db_conn:
                    return g.db_conn
            else:
                return g.db_conn
        except Exception as e:
            logger.warning(f"⚠️ Database connection attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    logger.error("❌ Failed to get database connection after retries")
    return None

def return_db_connection(exception=None):
    """Return database connection to the pool"""
    conn = g.pop('db_conn', None)
    if conn is not None and db_pool:
        db_pool.putconn(conn)

@app.teardown_appcontext
def close_db_connection(e):
    """Close database connection after each request"""
    return_db_connection()

# Function to execute database operations safely
def execute_db_query(query, params=None, fetch=True):
    """Execute a database query with proper connection handling"""
    conn = get_db_connection()
    if not conn:
        logger.error("❌ No database connection available")
        return None
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                result = cur.fetchall()
            else:
                result = None
                conn.commit()
            return result
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ Database error: {str(e)}")
        return None

# Cache for vector search results to reduce Pinecone calls
@lru_cache(maxsize=100)
def cached_vector_search(query_text_hash):
    """Cached wrapper for vector search - reduces Pinecone API calls"""
    query_text = query_text_hash  # Just using the hash parameter pattern for LRU cache
    
    try:
        query_response = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text
        )
        query_embedding = query_response["embedding"]

        search_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        context_items = []
        for match in search_results["matches"]:
            if "metadata" in match and "text" in match["metadata"]:
                context_items.append({
                    "text": match["metadata"]["text"],
                    "score": match["score"]
                })
                
        return context_items
    except Exception as e:
        logger.error(f"❌ Error in vector search: {str(e)}")
        return []

def get_context_for_query(query_text):
    """Get context for a query with parallel processing"""
    # First check if we have an exact cache hit
    cache_key = query_text
    if cache_key in response_cache:
        logger.info("✅ Cache hit for exact query")
        return response_cache[cache_key]['context']
    
    # Otherwise perform vector search
    try:
        context_items = cached_vector_search(query_text)
        
        if not context_items:
            return "No relevant information found."
            
        # Format context text
        context_texts = [item["text"] for item in context_items]
        context = "\n-----------------\n".join(context_texts)
        
        # Cache the context
        if cache_key not in response_cache:
            response_cache[cache_key] = {}
        response_cache[cache_key]['context'] = context
        
        return context
    except Exception as e:
        logger.error(f"❌ Error getting context: {str(e)}")
        return "Error retrieving context information."

def generate_llm_response(query, context):
    """Generate response using LLM with optimized prompt"""
    try:
        # Check if we already have this response cached
        cache_key = query
        if cache_key in response_cache and 'response' in response_cache[cache_key]:
            logger.info("✅ Cache hit for LLM response")
            return response_cache[cache_key]['response']
        
        # Create an optimized prompt for voice interaction
        prompt = f"""
You are a helpful voice assistant for Tecnvirons. Your name is Ninja Genie. Keep your answers brief, conversational, and suitable for voice.

Question: {query}

Context: {context}

Respond in 1-3 sentences maximum. Be direct and conversational. Remember you're speaking, not writing.
"""
        # Generate response
        response = llm_model_gemini.generate_content(prompt)
        answer = response.text.strip()
        
        # Cache the response
        if cache_key not in response_cache:
            response_cache[cache_key] = {}
        response_cache[cache_key]['response'] = answer
        
        return answer
    except Exception as e:
        logger.error(f"❌ Error in LLM response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response. Please try again."

def get_ai_response(query):
    """Get AI response with parallel processing of context and answer generation"""
    if not query or not isinstance(query, str) or query.strip() == "":
        return "I didn't catch that. Could you please repeat your question?"
    
    try:
        # Start context retrieval in a separate thread
        context_future = executor.submit(get_context_for_query, query.strip())
        
        # Get context with timeout
        try:
            context = context_future.result(timeout=3.0)
        except TimeoutError:
            logger.warning("⚠️ Context retrieval timeout, using fallback")
            context = "Unable to retrieve specific information in time."
        
        # Generate response
        answer = generate_llm_response(query.strip(), context)
        return answer
    except Exception as e:
        logger.error(f"❌ Error in get_ai_response: {str(e)}")
        return "I'm sorry, I couldn't process that request. Please try again."

# Store call state
class CallState:
    def __init__(self, call_uuid):
        self.call_uuid = call_uuid
        self.transcription_buffer = []
        self.last_response_time = time.time()
        self.conversation_history = []

# Routes
@app.route("/", methods=["GET"])
def home():
    return "✅ Voice Bot API is running!"

@app.route("/incoming-call", methods=["POST"])
def incoming_call():
    """Handle incoming call and present menu options"""
    try:
        call_uuid = request.form.get("CallUUID")
        caller_id = request.form.get("From")
        
        logger.info(f"📞 Incoming call from {caller_id} (UUID: {call_uuid})")
        
        # Store call state
        if call_uuid:
            active_calls[call_uuid] = CallState(call_uuid)
        
        # Pre-initialize services for this call in background
        threading.Thread(target=warm_services_for_call, args=(call_uuid,)).start()
        
        response = plivoxml.ResponseElement()
        get_digits = plivoxml.GetDigitsElement(
            action="https://web-production-aa492.up.railway.app/handle-menu",
            method="POST",
            timeout=5,  # Reduced from 10
            num_digits=1,
            retries=1
        )
        # Shorter welcome message for reduced latency
        get_digits.add_speak("Welcome to Tecnvirons. Press 1 to talk to Ninja Genie, our AI assistant. Press 2 to end the call.")
        response.add(get_digits)
        response.add_speak("No input received. Goodbye.")
        
        return Response(response.to_string(), mimetype='text/xml')
    except Exception as e:
        logger.error(f"❌ Error in incoming-call: {str(e)}")
        # Provide a fallback response
        response = plivoxml.ResponseElement()
        response.add_speak("We're experiencing technical difficulties. Please try again later.")
        return Response(response.to_string(), mimetype='text/xml')

def warm_services_for_call(call_uuid):
    """Pre-warm services for a specific call"""
    try:
        # Perform a dummy query to ensure services are ready
        _ = llm_model_gemini.generate_content("Hello")
        logger.info(f"✅ Services pre-warmed for call {call_uuid}")
    except Exception as e:
        logger.error(f"❌ Error pre-warming services for call {call_uuid}: {str(e)}")

@app.route("/handle-menu", methods=["POST"])
def handle_menu():
    """Process menu selection from user"""
    try:
        digit = request.form.get("Digits")
        call_uuid = request.form.get("CallUUID")
        
        logger.info(f"📲 Menu Selection: {digit} (UUID: {call_uuid})")
        
        response = plivoxml.ResponseElement()
        
        if digit == "1":
            # Start conversation with AI assistant
            welcome_text = "I'm Ninja Genie, how can I help you today? Please speak after the beep."
            response.add(plivoxml.SpeakElement(welcome_text))
            
            # Configure recording with shorter timeout for better conversation flow
            response.add(
                plivoxml.RecordElement(
                    action="https://web-production-aa492.up.railway.app/process-recording",
                    method="POST",
                    max_length=20,  # Reduced from 30 for more conversational feel
                    timeout=3,      # Reduced from 5 for faster response
                    transcription_type="auto",
                    transcription_url="https://web-production-aa492.up.railway.app/transcription",
                    transcription_method="POST",
                    play_beep=True
                )
            )
        else:
            response.add(plivoxml.SpeakElement("Thank you for calling. Goodbye."))
            response.add(plivoxml.HangupElement())
        
        return Response(response.to_string(), mimetype='text/xml')
    except Exception as e:
        logger.error(f"❌ Error in handle-menu: {str(e)}")
        # Provide a fallback response
        response = plivoxml.ResponseElement()
        response.add_speak("We're experiencing technical difficulties. Please try again later.")
        return Response(response.to_string(), mimetype='text/xml')

@app.route("/transcription", methods=["POST"])
def save_transcription():
    """Process and save transcription from Plivo"""
    try:
        content_type = request.headers.get('Content-Type', '')
        logger.info(f"📄 Transcription received - Content-Type: {content_type}")
        
        recording_id = None
        transcription_text = None
        call_uuid = request.form.get("CallUUID") or request.get_json(silent=True).get("CallUUID") if request.is_json else None
        
        # Handle different input formats
        if 'application/json' in content_type:
            data = request.get_json(silent=True) or {}
            recording_id = data.get("recording_id") or data.get("RecordingID")
            transcription_text = data.get("transcription") or data.get("TranscriptionText")
            call_uuid = call_uuid or data.get("CallUUID")
        else:
            form_data = request.form.to_dict()
            recording_id = form_data.get("RecordingID") or form_data.get("recording_id")
            transcription_text = form_data.get("TranscriptionText") or form_data.get("transcription")
            call_uuid = call_uuid or form_data.get("CallUUID")
        
        # Fallback to raw data if needed
        if not recording_id or not transcription_text:
            try:
                raw_data = request.get_data().decode('utf-8')
                if raw_data.startswith('{') and raw_data.endswith('}'):
                    json_data = json.loads(raw_data)
                    recording_id = recording_id or json_data.get("recording_id") or json_data.get("RecordingID")
                    transcription_text = transcription_text or json_data.get("transcription") or json_data.get("TranscriptionText")
                    call_uuid = call_uuid or json_data.get("CallUUID")
            except Exception as e:
                logger.error(f"❌ Error processing raw data: {str(e)}")
        
        if recording_id and transcription_text:
            recording_id_str = str(recording_id)
            clean_transcript = transcription_text.strip()
            
            # Store in memory cache for fast retrieval
            transcription_cache[recording_id_str] = clean_transcript
            
            # If we have the call_uuid, update the call state
            if call_uuid and call_uuid in active_calls:
                active_calls[call_uuid].transcription_buffer.append(clean_transcript)
            
            logger.info(f"✅ Saved transcription for Recording ID: {recording_id_str}")
            
            # Pre-fetch AI response in background to reduce latency
            threading.Thread(target=prefetch_ai_response, args=(clean_transcript, recording_id_str)).start()
            
            return "OK", 200
        else:
            logger.warning("⚠️ Missing Recording ID or Transcription Text")
            return "Missing required fields", 400
    except Exception as e:
        logger.error(f"❌ Error in save_transcription: {str(e)}")
        return "Error processing transcription", 500

def prefetch_ai_response(query, recording_id):
    """Pre-fetch AI response to reduce latency"""
    try:
        if query and query.strip():
            # Get AI response and store in cache
            response = get_ai_response(query)
            response_cache[recording_id] = {'query': query, 'response': response, 'timestamp': time.time()}
            logger.info(f"✅ Pre-fetched AI response for Recording ID: {recording_id}")
    except Exception as e:
        logger.error(f"❌ Error pre-fetching AI response: {str(e)}")

@app.route("/process-recording", methods=["POST"])
def process_recording():
    """Process recording and respond to user query"""
    try:
        recording_url = request.form.get("RecordUrl")
        recording_id = request.form.get("RecordingID")
        call_uuid = request.form.get("CallUUID")
        
        logger.info(f"🎙️ Processing recording: {recording_id} (Call UUID: {call_uuid})")
        
        recording_id_str = str(recording_id) if recording_id else None
        
        # Check if we already have a pre-fetched response
        if recording_id_str in response_cache:
            logger.info(f"✅ Using pre-fetched response for recording {recording_id_str}")
            reply = response_cache[recording_id_str]['response']
        else:
            # Try to get the transcription with reduced waiting
            transcript = None
            max_attempts = 6  # Reduced from 12
            wait_time = 0.5   # Reduced from 1.5
            
            logger.info(f"⏳ Checking for transcription: {recording_id_str}")
            
            # First check our fast cache
            if recording_id_str in transcription_cache:
                transcript = transcription_cache[recording_id_str]
                logger.info(f"✅ Found transcription in memory cache")
            
            # If not in cache, wait and check for it
            if not transcript:
                for attempt in range(max_attempts):
                    if recording_id_str in transcription_cache:
                        transcript = transcription_cache[recording_id_str]
                        logger.info(f"✅ Transcription found on attempt {attempt + 1}")
                        break
                    
                    # Try database as fallback (less frequently)
                    if attempt % 3 == 2:
                        try:
                            query = "SELECT transcription_text FROM transcriptions WHERE recording_id = %s ORDER BY created_at DESC LIMIT 1"
                            result = execute_db_query(query, (recording_id_str,))
                            if result and result[0]:
                                transcript = result[0][0]
                                logger.info(f"✅ Transcription found in database")
                                break
                        except Exception as e:
                            logger.error(f"❌ Database lookup error: {str(e)}")
                    
                    logger.info(f"⏳ Waiting for transcription: attempt {attempt + 1}/{max_attempts}")
                    time.sleep(wait_time)
            
            # If no transcription found, use a fallback
            if not transcript:
                logger.warning("⚠️ No transcription received, using fallback")
                transcript = "I didn't catch that clearly. Could you please repeat?"
                reply = "I'm sorry, I didn't catch what you said. Could you please repeat your question?"
            else:
                # Update call history if available
                if call_uuid and call_uuid in active_calls:
                    active_calls[call_uuid].conversation_history.append({
                        "role": "user",
                        "text": transcript
                    })
                
                # Get AI response
                reply = get_ai_response(transcript)
                
                # Update conversation history
                if call_uuid and call_uuid in active_calls:
                    active_calls[call_uuid].conversation_history.append({
                        "role": "assistant",
                        "text": reply
                    })
        
        logger.info(f"🤖 Responding with: {reply[:50]}...")
        
        # Build response XML
        response = plivoxml.ResponseElement()
        response.add(plivoxml.SpeakElement(reply))
        
        # Continue conversation
        response.add(plivoxml.SpeakElement("Do you have another question? Please speak after the beep."))
        response.add(plivoxml.RecordElement(
            action="https://web-production-aa492.up.railway.app/process-recording",
            method="POST",
            max_length=20,
            timeout=3,  # Reduced for faster conversation
            transcription_type="auto",
            transcription_url="https://web-production-aa492.up.railway.app/transcription",
            transcription_method="POST",
            play_beep=True
        ))
        
        return Response(response.to_string(), mimetype="text/xml")
    except Exception as e:
        logger.error(f"❌ Error in process_recording: {str(e)}")
        # Fallback response
        response = plivoxml.ResponseElement()
        response.add(plivoxml.SpeakElement("I'm sorry, we're experiencing technical difficulties. Let me try again. What was your question?"))
        response.add(plivoxml.RecordElement(
            action="https://web-production-aa492.up.railway.app/process-recording",
            method="POST",
            max_length=20,
            timeout=3,
            transcription_type="auto",
            transcription_url="https://web-production-aa492.up.railway.app/transcription",
            transcription_method="POST",
            play_beep=True
        ))
        return Response(response.to_string(), mimetype="text/xml")

# Call cleanup route - could be triggered by Plivo hangup webhook
@app.route("/call-ended", methods=["POST"])
def call_ended():
    """Handle call termination and cleanup resources"""
    try:
        call_uuid = request.form.get("CallUUID")
        logger.info(f"📞 Call ended: {call_uuid}")
        
        # Clean up resources
        if call_uuid in active_calls:
            del active_calls[call_uuid]
            
        return "OK", 200
    except Exception as e:
        logger.error(f"❌ Error in call_ended: {str(e)}")
        return "Error", 500

# Periodic cleanup function to run in background
def periodic_cleanup():
    """Clean up old cache entries"""
    while True:
        try:
            current_time = time.time()
            # Clean up response cache (keep for 30 minutes)
            expired_keys = [k for k, v in response_cache.items() 
                           if 'timestamp' in v and current_time - v['timestamp'] > 1800]
            for key in expired_keys:
                del response_cache[key]
                
            # Clean up transcription cache (keep for 10 minutes)
            expired_transcriptions = []
            for k, v in transcription_cache.items():
                if k in expired_keys:
                    expired_transcriptions.append(k)
            
            for key in expired_transcriptions:
                del transcription_cache[key]
                
            # Clean up inactive calls (after 30 minutes)
            expired_calls = [k for k, v in active_calls.items() 
                            if current_time - v.last_response_time > 1800]
            for key in expired_calls:
                del active_calls[key]
                
            logger.info(f"🧹 Cleaned up {len(expired_keys)} cache entries, {len(expired_calls)} inactive calls")
        except Exception as e:
            logger.error(f"❌ Error in periodic cleanup: {str(e)}")
        
        # Run every 5 minutes
        time.sleep(300)

# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

# Graceful shutdown function
def shutdown_handler():
    """Close all connections and clean up resources on shutdown"""
    global db_pool
    if db_pool is not None:
        db_pool.closeall()
        logger.info("✅ All database connections closed during shutdown")
    
    # Allow executor to complete running tasks
    executor.shutdown(wait=False)
    logger.info("✅ Thread executor shutdown")

# Register shutdown handler
import atexit
atexit.register(shutdown_handler)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Use threaded=True for better handling of concurrent requests
    socketio.run(app, host="0.0.0.0", port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)