import os
import time
import json
import logging
import asyncio
import threading
import uvicorn
from queue import Queue
from typing import Dict, Optional, List
import wave
import io
import base64
import uuid
from pydantic import BaseModel
from contextlib import contextmanager

# Flask and FastAPI imports
from flask import Flask, request, Response
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.wsgi import WSGIMiddleware

# Database imports
import psycopg2
from psycopg2.pool import SimpleConnectionPool

# Plivo imports
from plivo import plivoxml

# AI and vector search imports
import openai
from realtime import RealtimeClient
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# Environment variables
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app for Plivo webhooks
flask_app = Flask(__name__)

# Initialize FastAPI app for WebSockets/async operations
fastapi_app = FastAPI()

# Add Flask app as a middleware to FastAPI
fastapi_app.mount("/plivo", WSGIMiddleware(flask_app))

# Global variables
db_pool = None
pinecone_index = None
active_calls: Dict[str, Dict] = {}  # Track active calls by call_uuid

# ======================== DATABASE CONNECTION MANAGEMENT ========================
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = db_pool.getconn()
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            db_pool.putconn(conn)

def execute_db_query(query, params=None, fetch=True):
    """Execute a database query with proper connection handling"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if fetch:
                    result = cur.fetchall()
                else:
                    result = None
                    conn.commit()
                return result
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return None

def init_db_pool():
    """Initialize the database connection pool"""
    global db_pool
    try:
        if db_pool is None:
            db_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=os.getenv('DATABASE_URL'),
                sslmode='require'
            )
            logger.info("✅ Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {str(e)}")
        raise

# ======================== USER DATA FUNCTIONS ========================
def get_user_by_phone(phone_number):
    """Get user by phone number"""
    try:
        normalized_phone = ''.join(filter(str.isdigit, str(phone_number)))
        query = """
            SELECT username, email
            FROM users
            WHERE phone_number = %s
            LIMIT 1;
        """
        result = execute_db_query(query, (normalized_phone,))
        return {'username': result[0][0], 'email': result[0][1]} if result and result[0] else None
    except Exception as e:
        logger.error(f"Error in get_user_by_phone: {str(e)}")
        return None

def store_user_details(username, email, phone_number):
    """Store or update user details in database"""
    try:
        normalized_phone = ''.join(filter(str.isdigit, str(phone_number)))
        # Check if user exists
        query = "SELECT id FROM users WHERE phone_number = %s"
        result = execute_db_query(query, (normalized_phone,))
        
        if result and result[0]:
            # Update existing user
            query = """
                UPDATE users 
                SET username = %s, email = %s 
                WHERE phone_number = %s
            """
            execute_db_query(query, (username, email, normalized_phone), fetch=False)
            logger.info(f"✅ Updated user details for {phone_number}")
        else:
            # Insert new user
            query = """
                INSERT INTO users (username, email, phone_number) 
                VALUES (%s, %s, %s)
            """
            execute_db_query(query, (username, email, normalized_phone), fetch=False)
            logger.info(f"✅ Stored new user details for {phone_number}")
        return True
    except Exception as e:
        logger.error(f"Error in store_user_details: {str(e)}")
        return False

# ======================== VECTOR SEARCH AND AI FUNCTIONS ========================
def validate_environment():
    """Validate all required environment variables and API keys"""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for GPT-4 and Realtime API",
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

def init_ai_services():
    """Initialize AI services (OpenAI, Gemini, Pinecone)"""
    global pinecone_index
    
    try:
        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        logger.info("✅ OpenAI API configured")
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv("GENAI_API_KEY"))
        if not os.getenv("GENAI_API_KEY"):
            raise ValueError("GENAI_API_KEY not found in environment variables")
        logger.info("✅ Gemini API configured")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY not found in environment variables")
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
            
        pinecone_index = pc.Index(index_name)
        logger.info("✅ Connected to Pinecone index")
        
    except Exception as e:
        logger.error(f"❌ Error during AI service initialization: {str(e)}")
        raise

def generate_vector_search_answer(query_text, namespace="default"):
    """Generate answer using vector search with error handling"""
    try:
        if not pinecone_index:
            raise ValueError("Pinecone index not initialized")
            
        logger.info("🔍 Generating embeddings...")
        query_response = genai.embed_content(
            model="models/text-embedding-004",
            content=query_text
        )
        query_embedding = query_response["embedding"]

        logger.info("🔍 Searching in Pinecone...")
        search_results = pinecone_index.query(
            vector=query_embedding,
            namespace=namespace,
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
        logger.error(f"❌ Error in generate_vector_search_answer: {str(e)}")
        return "I apologize, but I'm having trouble searching for information. Please try again."

def request_llm_to_get_summarize(query, context):
    """Generate response using Gemini with error handling"""
    try:
        logger.info(f"🤖 Processing query: {query[:100]}...")
        llm_model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        
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

# ======================== OPENAI REALTIME CLIENT FUNCTIONS ========================
async def create_realtime_client(call_uuid, caller_id):
    """Create and configure a new OpenAI Realtime client for a call"""
    try:
        logger.info(f"Creating Realtime client for call {call_uuid}")
        realtime_client = RealtimeClient(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Set up event handlers
        realtime_client.on("conversation.updated", lambda event: handle_conversation_updated(event, call_uuid))
        realtime_client.on("conversation.item.completed", lambda item: handle_item_completed(item, call_uuid))
        realtime_client.on("conversation.interrupted", lambda event: handle_conversation_interrupt(event, call_uuid))
        realtime_client.on("error", lambda event: handle_error(event, call_uuid))
        
        # Connect to the OpenAI Realtime API
        await realtime_client.connect()
        
        # Store client in active_calls
        if call_uuid not in active_calls:
            active_calls[call_uuid] = {}
        
        active_calls[call_uuid]["realtime_client"] = realtime_client
        active_calls[call_uuid]["caller_id"] = caller_id
        active_calls[call_uuid]["audio_queue"] = Queue()
        active_calls[call_uuid]["transcript"] = ""
        active_calls[call_uuid]["messages"] = [
            {"role": "system", "content": get_initial_system_prompt()},
        ]
        
        # Add tools
        await asyncio.gather(
            realtime_client.add_tool(get_vector_search_answer_def(), get_vector_search_answer_handler(call_uuid)),
            realtime_client.add_tool(check_user_by_phone_in_db_def(), check_user_by_phone_in_db_handler(call_uuid)),
            realtime_client.add_tool(store_user_details_in_db_def(), store_user_details_in_db_handler(call_uuid))
        )
        
        logger.info(f"✅ Realtime client created and connected for call {call_uuid}")
        return realtime_client
    except Exception as e:
        logger.error(f"❌ Error creating Realtime client: {str(e)}")
        return None

async def handle_conversation_updated(event, call_uuid):
    """Handle streaming responses from the Realtime API"""
    if call_uuid not in active_calls:
        logger.warning(f"Received event for unknown call {call_uuid}")
        return
    
    delta = event.get("delta")
    if not delta:
        return
        
    # Handle audio streaming
    if "audio" in delta:
        audio = delta["audio"]  # Int16Array audio data
        if "audio_buffer" not in active_calls[call_uuid]:
            active_calls[call_uuid]["audio_buffer"] = []
        active_calls[call_uuid]["audio_buffer"].extend(audio)
        
    # Handle transcript streaming
    if "transcript" in delta:
        transcript = delta["transcript"]  # string, transcript added
        if "current_transcript" not in active_calls[call_uuid]:
            active_calls[call_uuid]["current_transcript"] = ""
        active_calls[call_uuid]["current_transcript"] += transcript
        logger.debug(f"Current transcript for {call_uuid}: {active_calls[call_uuid]['current_transcript']}")

async def handle_item_completed(item, call_uuid):
    """Process completed conversation items"""
    if call_uuid not in active_calls:
        logger.warning(f"Received item completed for unknown call {call_uuid}")
        return
        
    item_type = item.get("type")
    if item_type == "message":
        role = item.get("role")
        content = item.get("content", "")
        
        # Add message to conversation history
        if "messages" not in active_calls[call_uuid]:
            active_calls[call_uuid]["messages"] = []
            
        active_calls[call_uuid]["messages"].append({"role": role, "content": content})
        
        # If this was an assistant message, send audio to Plivo
        if role == "assistant":
            active_calls[call_uuid]["transcript"] = content
            logger.info(f"Assistant response for {call_uuid}: {content[:100]}...")
            
            # Flush the audio buffer for streaming to Plivo
            if "audio_buffer" in active_calls[call_uuid] and active_calls[call_uuid]["audio_buffer"]:
                # Convert audio buffer to proper format and queue for streaming
                audio_data_to_queue = active_calls[call_uuid]["audio_buffer"]
                active_calls[call_uuid]["audio_queue"].put(audio_data_to_queue)
                active_calls[call_uuid]["audio_buffer"] = []

async def handle_conversation_interrupt(event, call_uuid):
    """Handle conversation interruption"""
    if call_uuid not in active_calls:
        return
    logger.info(f"Conversation interrupted for call {call_uuid}")
    # Clear current audio being processed
    if "audio_buffer" in active_calls[call_uuid]:
        active_calls[call_uuid]["audio_buffer"] = []

async def handle_error(event, call_uuid):
    """Handle errors from the Realtime API"""
    logger.error(f"Realtime API error for call {call_uuid}: {event}")

def get_initial_system_prompt():
    """Get the initial system prompt for the OpenAI Realtime client"""
    return """
You are a helpful agent designed to assist users with inquiries and store their details. Follow these steps:
1. **Collect User Details**:
   - At the start of the chat, ask the user for their **Phone Number** and call the tool check_user_by_phone_in_db that determines if the user already present.
   - If the user present, give details found in the db Name, Email to the user, wish them and ask them to ask inquiries about the Product.
   - If the user not present in the database, ask the user for their **Name** and **Email ID**.
      - If any detail is missing, politely ask for it. Only proceed when all three details are provided.
      - **Store User Details**:
        - Once all details are collected, call the tool store_user_details_in_db to save the information.
        - Confirm to the user that their details have been stored.

2. **Handle Updates**:
   - If the user requests changes to their details, call store_user_details_in_db with the updated information and inform the user.

3. **Assist with Inventory Inquiries**:
   - After storing details, ask the user about their product inquiries.
   - For each inquiry, call the tool get_vector_search_answer to retrieve the answer.
   - Summarize the answer and respond to the user. Encourage them to ask more questions.

Note:
- If the user input inquiry in Non-English language, you should first convert it into 'English' and send it to 'get_vector_search_answer'.
Always be polite, clear, and concise in your responses.
"""

# ======================== TOOL DEFINITIONS ========================
def get_vector_search_answer_def():
    """Define the vector search tool"""
    return {
        "name": "get_vector_search_answer",
        "description": "Get answers to product-related questions using vector search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to answer"
                }
            },
            "required": ["query"]
        }
    }

def get_vector_search_answer_handler(call_uuid):
    """Create a handler for vector search tool"""
    async def handler(params):
        query = params.get("query", "")
        if query:
            answer = generate_vector_search_answer(query)
            return answer
        return "I couldn't process that question. Please try again."
    return handler

def check_user_by_phone_in_db_def():
    """Define the check user by phone tool"""
    return {
        "name": "check_user_by_phone_in_db",
        "description": "Check if a user exists in the database by phone number",
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "The phone number to check"
                }
            },
            "required": ["phone_number"]
        }
    }

def check_user_by_phone_in_db_handler(call_uuid):
    """Create a handler for checking user by phone"""
    async def handler(params):
        phone_number = params.get("phone_number", "")
        if not phone_number:
            return {"exists": False, "message": "No phone number provided"}
            
        # Use caller ID if available and no phone number provided
        if phone_number == "caller" and call_uuid in active_calls:
            phone_number = active_calls[call_uuid].get("caller_id", "")
            
        user_data = get_user_by_phone(phone_number)
        if user_data:
            return {
                "exists": True,
                "username": user_data["username"],
                "email": user_data["email"],
                "message": "User found in database"
            }
        else:
            return {"exists": False, "message": "User not found in database"}
    return handler

def store_user_details_in_db_def():
    """Define the store user details tool"""
    return {
        "name": "store_user_details_in_db",
        "description": "Store user details in the database",
        "parameters": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "The user's name"
                },
                "email": {
                    "type": "string",
                    "description": "The user's email address"
                },
                "phone_number": {
                    "type": "string",
                    "description": "The user's phone number or 'caller' to use the caller ID"
                }
            },
            "required": ["username", "email", "phone_number"]
        }
    }

def store_user_details_in_db_handler(call_uuid):
    """Create a handler for storing user details"""
    async def handler(params):
        username = params.get("username", "")
        email = params.get("email", "")
        phone_number = params.get("phone_number", "")
        
        if phone_number == "caller" and call_uuid in active_calls:
            phone_number = active_calls[call_uuid].get("caller_id", "")
        
        if not all([username, email, phone_number]):
            return {"success": False, "message": "Missing required user details"}
            
        success = store_user_details(username, email, phone_number)
        if success:
            return {"success": True, "message": "User details stored successfully"}
        else:
            return {"success": False, "message": "Failed to store user details"}
    return handler

# ======================== AUDIO PROCESSING FUNCTIONS ========================
def pcm_to_wav(pcm_data, sample_rate=24000):
    """Convert PCM audio data to WAV format"""
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_io.getvalue()

def process_audio_for_plivo(audio_data):
    """Process audio data for Plivo"""
    # Convert int16 array to bytes
    audio_bytes = bytes(bytearray(audio_data))
    # Convert to wav format
    wav_bytes = pcm_to_wav(audio_bytes)
    # Encode as base64 for Plivo
    return base64.b64encode(wav_bytes).decode('utf-8')

# ======================== PLIVO WEBHOOK ROUTES ========================
@flask_app.route("/", methods=["GET"])
def home():
    return "✅ Combined Plivo and OpenAI Realtime Voice Bot is running!"

@flask_app.route("/incoming-call", methods=["POST"])
def incoming_call():
    """Handle incoming calls from Plivo"""
    try:
        logger.info(f"📞 Incoming call: {request.form}")
        
        call_uuid = request.form.get("CallUUID")
        caller_id = request.form.get("From")
        
        if not call_uuid or not caller_id:
            logger.error("Missing required call parameters")
            return Response("Missing required parameters", status=400)
        
        # Create a response with Plivo XML
        response = plivoxml.ResponseElement()
        response.add_speak("Welcome to our product information service. Please ask about any product you're interested in.")
        
        # Add WebSocket connection for real-time audio
        connect = response.add_connect()
        stream = connect.add_stream()
        stream.set_attributes({
            "callbackUrl": f"wss://web-production-aa492.up.railway.app/ws/call/{call_uuid}",
            "callbackMethod": "GET",
            "streamTimeout": "600",  # 10 minutes max call time
            "contentType": "audio/l16;rate=16000,audio/l16;rate=8000"
        })
        
        # Store call information
        active_calls[call_uuid] = {
            "from_number": caller_id,
            "start_time": time.time(),
            "audio_queue": Queue(),
            "transcript": "",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides information about products and services."}
            ]
        }
        
        # Create Realtime client in a separate thread
        asyncio.run_coroutine_threadsafe(
            create_realtime_client(call_uuid, caller_id),
            asyncio.get_event_loop()
        )
        
        logger.info(f"Call answered: {call_uuid} from {caller_id}")
        return Response(str(response), mimetype="text/xml")
        
    except Exception as e:
        logger.error(f"Error in incoming_call: {str(e)}")
        return Response("Internal Server Error", status=500)

@flask_app.route("/hangup", methods=["POST"])
def call_hangup():
    """Handle call hangup events"""
    call_uuid = request.form.get("CallUUID")
    logger.info(f"📞 Call hangup: {call_uuid}")
    
    if call_uuid in active_calls:
        # Cleanup resources
        if "realtime_client" in active_calls[call_uuid]:
            asyncio.run_coroutine_threadsafe(
                active_calls[call_uuid]["realtime_client"].disconnect(),
                asyncio.get_event_loop()
            )
        active_calls.pop(call_uuid, None)
    
    return "OK"

@flask_app.route("/answer", methods=["GET", "POST"])
def answer_call():
    """Handle incoming calls"""
    try:
        # Get call parameters
        call_uuid = request.values.get("CallUUID")
        from_number = request.values.get("From")
        
        if not call_uuid or not from_number:
            logger.error("Missing required call parameters")
            return Response("Missing required parameters", status=400)
            
        # Create response
        response = plivoxml.Response()
        speak = response.addSpeak("Hello! How can I help you today?")
        speak.setAttributes({
            "voice": "Polly.Amy",
            "language": "en-GB"
        })
        
        # Add record element
        record = response.addRecord()
        record.setAttributes({
            "action": f"/recording_callback/{call_uuid}",
            "method": "POST",
            "maxLength": "30",
            "playBeep": "true",
            "finishOnKey": "#",
            "transcriptionType": "auto",
            "transcriptionUrl": f"/transcription_callback/{call_uuid}",
            "transcriptionMethod": "POST"
        })
        
        # Store call information
        active_calls[call_uuid] = {
            "from_number": from_number,
            "start_time": time.time()
        }
        
        logger.info(f"Call answered: {call_uuid} from {from_number}")
        return Response(str(response), mimetype="text/xml")
        
    except Exception as e:
        logger.error(f"Error in answer_call: {str(e)}")
        return Response("Internal Server Error", status=500)

@flask_app.route("/recording_callback/<call_uuid>", methods=["POST"])
def recording_callback(call_uuid):
    """Handle recording completion"""
    try:
        if call_uuid not in active_calls:
            logger.error(f"Unknown call UUID: {call_uuid}")
            return Response("Unknown call", status=404)
            
        recording_url = request.values.get("RecordUrl")
        if not recording_url:
            logger.error("No recording URL provided")
            return Response("No recording URL", status=400)
            
        # Store recording URL
        active_calls[call_uuid]["recording_url"] = recording_url
        logger.info(f"Recording received for call {call_uuid}")
        
        return Response("OK", status=200)
        
    except Exception as e:
        logger.error(f"Error in recording_callback: {str(e)}")
        return Response("Internal Server Error", status=500)

@flask_app.route("/transcription_callback/<call_uuid>", methods=["POST"])
def transcription_callback(call_uuid):
    """Handle transcription completion"""
    try:
        if call_uuid not in active_calls:
            logger.error(f"Unknown call UUID: {call_uuid}")
            return Response("Unknown call", status=404)
            
        # Get transcription data
        transcription = request.values.get("transcription")
        if not transcription:
            logger.error("No transcription provided")
            return Response("No transcription", status=400)
            
        # Store transcription
        active_calls[call_uuid]["transcription"] = transcription
        logger.info(f"Transcription received for call {call_uuid}: {transcription}")
        
        # Process transcription and generate response
        response = get_ai_response(transcription)
        
        # Create Plivo response
        plivo_response = plivoxml.Response()
        speak = plivo_response.addSpeak(response)
        speak.setAttributes({
            "voice": "Polly.Amy",
            "language": "en-GB"
        })
        
        return Response(str(plivo_response), mimetype="text/xml")
        
    except Exception as e:
        logger.error(f"Error in transcription_callback: {str(e)}")
        return Response("Internal Server Error", status=500)

# ======================== AI RESPONSE GENERATION ========================
def get_ai_response(query: str) -> str:
    """Generate AI response based on user query"""
    try:
        # First try to get product details from database
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Search for exact match
                cur.execute("""
                    SELECT product_name, product_quantity, product_rate, product_value, description 
                    FROM products 
                    WHERE LOWER(product_name) = LOWER(%s)
                """, (query,))
                result = cur.fetchone()
                
                if result:
                    product_name, quantity, rate, value, description = result
                    return f"""
                    I found information about {product_name}:
                    - Available Quantity: {quantity}
                    - Price per Unit: ${rate}
                    - Total Value: ${value}
                    - Description: {description}
                    Would you like to know more about this product?
                    """
                
                # If no exact match, search for similar products
                cur.execute("""
                    SELECT product_name, product_quantity, product_rate, product_value, description 
                    FROM products 
                    WHERE LOWER(product_name) LIKE LOWER(%s)
                """, (f"%{query}%",))
                result = cur.fetchone()
                
                if result:
                    product_name, quantity, rate, value, description = result
                    return f"""
                    I found a similar product: {product_name}
                    - Available Quantity: {quantity}
                    - Price per Unit: ${rate}
                    - Total Value: ${value}
                    - Description: {description}
                    Is this the product you were looking for?
                    """
        
        # If no product found, use GPT-4 for general response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides information about products and services."},
                {"role": "user", "content": query}
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error in get_ai_response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again later."

# ======================== ERROR HANDLERS ========================
@flask_app.errorhandler(404)
def not_found_error(error):
    return Response("Not Found", status=404)

@flask_app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal Server Error: {str(error)}")
    return Response("Internal Server Error", status=500)

@fastapi_app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global Exception: {str(exc)}")
    return Response("Internal Server Error", status=500)

# ======================== WEBSOCKET HANDLER ========================
@fastapi_app.websocket("/ws/call/{call_uuid}")
async def websocket_endpoint(websocket: WebSocket, call_uuid: str):
    await websocket.accept()
    logger.info(f"WebSocket connection established for call {call_uuid}")
    
    if call_uuid not in active_calls:
        logger.error(f"No active call found for {call_uuid}")
        await websocket.close()
        return
    
    # Start the greeting
    realtime_client = active_calls[call_uuid].get("realtime_client")
    if not realtime_client:
        logger.error(f"No Realtime client for call {call_uuid}")
        await websocket.close()
        return
    
    # Create tasks for bidirectional audio streaming
    receive_task = asyncio.create_task(receive_audio(websocket, call_uuid))
    send_task = asyncio.create_task(send_audio(websocket, call_uuid))
    
    try:
        # Run both tasks concurrently
        await asyncio.gather(receive_task, send_task)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for call {call_uuid}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {str(e)}")
    finally:
        # Clean up and disconnect
        if not receive_task.done():
            receive_task.cancel()
        if not send_task.done():
            send_task.cancel()
            
        # Disconnect the Realtime client
        if call_uuid in active_calls and "realtime_client" in active_calls[call_uuid]:
            await active_calls[call_uuid]["realtime_client"].disconnect()
            
        # Remove the call from active calls
        active_calls.pop(call_uuid, None)
        logger.info(f"Cleaned up resources for call {call_uuid}")

async def receive_audio(websocket: WebSocket, call_uuid: str):
    """Receive audio from Plivo and send to OpenAI Realtime"""
    realtime_client = active_calls[call_uuid].get("realtime_client")
    if not realtime_client:
        return
        
    try:
        while True:
            # Receive audio data from Plivo
            data = await websocket.receive_bytes()
            
            # Send audio to OpenAI Realtime
            await realtime_client.append_input_audio(data)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected while receiving audio for call {call_uuid}")
    except Exception as e:
        logger.error(f"Error receiving audio: {str(e)}")

async def send_audio(websocket: WebSocket, call_uuid: str):
    """Send audio from OpenAI Realtime to Plivo"""
    try:
        # Start with a greeting
        greeting = "Hello! I'm your AI assistant. May I have your phone number so I can look up your details?"
        active_calls[call_uuid]["messages"].append({"role": "assistant", "content": greeting})
        
        # Send initial greeting message as audio
        audio_response = await generate_speech(greeting)
        if audio_response:
            await websocket.send_text(json.dumps({
                "event": "media",
                "media": {
                    "payload": audio_response
                }
            }))
        
        # Then continuously check for and send new audio
        while True:
            # Check if there's audio in the queue
            if call_uuid in active_calls and "audio_queue" in active_calls[call_uuid]:
                queue = active_calls[call_uuid]["audio_queue"]
                
                if not queue.empty():
                    audio_data = queue.get()
                    processed_audio = process_audio_for_plivo(audio_data)
                    
                    # Send to Plivo
                    await websocket.send_text(json.dumps({
                        "event": "media",
                        "media": {
                            "payload": processed_audio
                        }
                    }))
            
            # Small delay to prevent CPU hogging
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected while sending audio for call {call_uuid}")
    except Exception as e:
        logger.error(f"Error sending audio: {str(e)}")

async def generate_speech(text):
    """Generate speech from text using OpenAI TTS"""
    try:
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        # Get the audio data
        audio_data = await response.read()
        # Convert to base64
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        return None

# ======================== APPLICATION STARTUP ========================
@fastapi_app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    try:
        # Validate environment variables
        validate_environment()
        
        # Initialize database
        init_db_pool()
        
        # Initialize AI services
        init_ai_services()
        
        logger.info("✅ Application started successfully")
    except Exception as e:
        logger.error(f"❌ Error during application startup: {str(e)}")
        raise

@fastapi_app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown"""
    global db_pool
    
    # Clean up active calls
    for call_uuid, call_data in active_calls.items():
        if "realtime_client" in call_data:
            try:
                await call_data["realtime_client"].disconnect()
            except:
                pass
    
    # Close database connections
    if db_pool:
        db_pool.closeall()
        logger.info("✅ All database connections closed")
        
    logger.info("✅ Application shut down successfully")

# Create a dedicated event loop for async operations
def init_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

def start_uvicorn():
    """Start Uvicorn server for FastAPI app"""
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    # Start event loop in a separate thread
    event_loop_thread = threading.Thread(target=lambda: init_event_loop().run_forever())
    event_loop_thread.daemon = True
    event_loop_thread.start()
    
    # Start Uvicorn server
    start_uvicorn()