from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import openai
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import json
import os
from datetime import datetime
import logging
import sqlite3
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')
    DATABASE_PATH = 'conversations.db'
    MAX_HISTORY_LENGTH = 10

config = Config()

# Initialize OpenAI
openai.api_key = config.OPENAI_API_KEY

# Database setup
def init_database():
    """Initialize SQLite database for storing conversations"""
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

# Initialize LangChain components
def create_conversation_chain():
    """Create a new conversation chain with memory"""
    try:
        # Initialize ChatOpenAI model - Using GPT-4o mini for better performance
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=config.OPENAI_API_KEY,
            max_tokens=1000
        )
        
        # Create memory for conversation
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )
        
        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )
        
        return conversation
        
    except Exception as e:
        logger.error(f"Error creating conversation chain: {str(e)}")
        return None

# Store conversation chains for each session
conversation_chains = {}

def get_or_create_conversation_chain(conversation_id):
    """Get existing conversation chain or create new one"""
    if conversation_id not in conversation_chains:
        conversation_chains[conversation_id] = create_conversation_chain()
        
        # Load previous conversation history
        load_conversation_history(conversation_id)
    
    return conversation_chains[conversation_id]

def load_conversation_history(conversation_id):
    """Load previous conversation history from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_message, assistant_response 
                FROM conversations 
                WHERE conversation_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (conversation_id, config.MAX_HISTORY_LENGTH))
            
            rows = cursor.fetchall()
            
            if rows and conversation_id in conversation_chains:
                chain = conversation_chains[conversation_id]
                # Add messages to memory in reverse order (oldest first)
                for user_msg, assistant_msg in reversed(rows):
                    chain.memory.chat_memory.add_user_message(user_msg)
                    chain.memory.chat_memory.add_ai_message(assistant_msg)
                    
    except Exception as e:
        logger.error(f"Error loading conversation history: {str(e)}")

def save_conversation(conversation_id, user_message, assistant_response):
    """Save conversation to database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (conversation_id, user_message, assistant_response)
                VALUES (?, ?, ?)
            ''', (conversation_id, user_message, assistant_response))
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")

def process_query_with_context(query, conversation_id):
    """Process user query with conversation context using LangChain"""
    try:
        # Get or create conversation chain
        conversation = get_or_create_conversation_chain(conversation_id)
        
        if not conversation:
            raise Exception("Failed to create conversation chain")
        
        # Add system message for better context (only for new conversations)
        if len(conversation.memory.chat_memory.messages) == 0:
            system_prompt = """You are a helpful AI voice assistant. You respond to user queries with informative and conversational answers. 
            Keep your responses clear, concise, and engaging. You can help with information about various topics, answer questions, 
            and have natural conversations. Always be polite and helpful."""
            
            conversation.memory.chat_memory.add_message(SystemMessage(content=system_prompt))
        
        # Process the query
        response = conversation.predict(input=query)
        
        # Save to database
        save_conversation(conversation_id, query, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"I apologize, but I encountered an error processing your request: {str(e)}"

# API Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Main chat endpoint for voice assistant"""
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        user_message = data['message'].strip()
        conversation_id = data.get('conversation_id', f'conv_{int(datetime.now().timestamp())}')
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            }), 400
        
        logger.info(f"Processing message: {user_message} (Conversation: {conversation_id})")
        
        # Process the query
        response = process_query_with_context(user_message, conversation_id)
        
        logger.info(f"Generated response: {response[:100]}...")
        
        return jsonify({
            'success': True,
            'response': response,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation_history(conversation_id):
    """Get conversation history for a specific conversation ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_message, assistant_response, timestamp 
                FROM conversations 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
            ''', (conversation_id,))
            
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    'user_message': row[0],
                    'assistant_response': row[1],
                    'timestamp': row[2]
                })
            
            return jsonify({
                'success': True,
                'conversation_id': conversation_id,
                'history': history
            })
            
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/conversations', methods=['GET'])
def get_all_conversations():
    """Get all conversation IDs"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT conversation_id, 
                       COUNT(*) as message_count,
                       MAX(timestamp) as last_message
                FROM conversations 
                GROUP BY conversation_id
                ORDER BY last_message DESC
            ''')
            
            rows = cursor.fetchall()
            
            conversations = []
            for row in rows:
                conversations.append({
                    'conversation_id': row[0],
                    'message_count': row[1],
                    'last_message': row[2]
                })
            
            return jsonify({
                'success': True,
                'conversations': conversations
            })
            
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/clear/<conversation_id>', methods=['DELETE'])
def clear_conversation(conversation_id):
    """Clear a specific conversation"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM conversations WHERE conversation_id = ?', (conversation_id,))
            conn.commit()
            
            # Remove from memory
            if conversation_id in conversation_chains:
                del conversation_chains[conversation_id]
            
            return jsonify({
                'success': True,
                'message': f'Conversation {conversation_id} cleared'
            })
            
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Initialize database on startup
init_database()

if __name__ == '__main__':
    # Check if OpenAI API key is set
    if config.OPENAI_API_KEY == 'your-openai-api-key-here':
        logger.warning("OpenAI API key not set! Please set the OPENAI_API_KEY environment variable.")
    
    logger.info("Starting AI Voice Assistant Backend...")
    logger.info(f"Database path: {config.DATABASE_PATH}")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
