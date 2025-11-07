import os
import sqlite3
import time
from flask import Flask, request, jsonify, g
from flask_cors import CORS 

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
# NEW: Import MessagesPlaceholder for conversation history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# -------------------------

# --- Configuration ---
class Config:
    """Application configuration settings."""
    
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b") 
    DATABASE_PATH = os.getenv("DATABASE_PATH", "ollama_chat_v2.db")
    LOG_LEVEL = 'INFO' 
    # NEW: Updated system prompt for a more capable agent
    LLM_SYSTEM_PROMPT = (
        "You are an expert-level personal assistant, skilled in all subjects. "
        "You will be given the current conversation history. "
        "Your responses must be engaging, informative, complete, and formatted in Markdown."
    )

# --- Database Management ---
def get_db_connection():
    """Establishes a connection to the SQLite database and uses 'g' for reuse."""
    if 'db' not in g:
        g.db = sqlite3.connect(Config.DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """Closes the database connection if it exists."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db(app):
    """Initializes the database schema."""
    with app.app_context():
        conn = get_db_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    app.logger.info(f"Database initialized at {Config.DATABASE_PATH}")


# --- LangChain LLM Setup ---
def initialize_llm_chain(app):
    """Initializes the Ollama LLM and the LangChain chain with memory."""
    try:
        llm = ChatOllama(model=Config.OLLAMA_MODEL)
        
        # NEW: The prompt now includes a placeholder for conversation history
        prompt = ChatPromptTemplate.from_messages([
            ("system", Config.LLM_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"), # For memory
            ("human", "{question}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        app.logger.info(f"Successfully connected to Ollama and model: {Config.OLLAMA_MODEL}")
        return chain
    except Exception as e:
        app.logger.error(f"FATAL: Failed to initialize ChatOllama or LangChain: {e}", exc_info=True)
        app.logger.error(f"Ensure Ollama is running and model '{Config.OLLAMA_MODEL}' is downloaded.")
        return None

# --- Application Factory ---
def create_app(test_config=None):
    """The application factory function."""
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app) 
    
    init_db(app)
    app.teardown_appcontext(close_db)
    
    app.llm_chain = initialize_llm_chain(app)

    @app.route('/status', methods=['GET'])
    def get_status():
        """Checks the readiness of the LLM chain."""
        # ... (same as your file)
        if app.llm_chain:
            return jsonify({
                "status": "ready", 
                "model": Config.OLLAMA_MODEL,
                "prompt": Config.LLM_SYSTEM_PROMPT
            }), 200
        else:
            return jsonify({"status": "error", "message": "LLM chain failed to initialize."}), 503

    @app.route('/ask', methods=['POST'])
    def ask_llm():
        """Handles new questions, incorporating chat history."""
        if not app.llm_chain:
            return jsonify({"error": f"LLM not ready."}), 503

        try:
            data = request.get_json()
            user_question = data.get('question')
            # NEW: Get the chat history from the request
            chat_history = data.get('history', [])
            
            if not user_question:
                return jsonify({"error": "Missing 'question' in request body"}), 400

            app.logger.info(f"Received question: {user_question}")

            # NEW: Format history for LangChain
            # React sends: [{'role': 'user', 'content': '...'}, {'role': 'ai', 'content': '...'}]
            # LangChain expects: [('human', '...'), ('ai', '...')]
            formatted_history = []
            for msg in chat_history:
                if msg.get('role') == 'user':
                    formatted_history.append(("human", msg.get('content')))
                elif msg.get('role') == 'ai':
                    formatted_history.append(("ai", msg.get('content')))

            # --- LLM Invocation with Memory ---
            start_time = time.time()
            llm_response = app.llm_chain.invoke({
                "chat_history": formatted_history,
                "question": user_question
            })
            end_time = time.time()
            # ----------------------------------

            # Save to database (only the *first* user question of a session)
            # We only save if there is no history, to prevent duplicates
            if not formatted_history:
                conn = get_db_connection()
                conn.execute(
                    "INSERT INTO interactions (question, answer) VALUES (?, ?)",
                    (user_question, llm_response)
                )
                conn.commit()

            app.logger.info(f"LLM Answer generated in {end_time - start_time:.2f}s")
            return jsonify({"answer": llm_response})

        except Exception as e:
            app.logger.error(f"Error during chain invocation: {e}", exc_info=True)
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @app.route('/history', methods=['GET'])
    def get_history():
        """Retrieves all past Q&A interactions, newest first."""
        # ... (same as your file)
        try:
            conn = get_db_connection()
            rows = conn.execute(
                "SELECT id, question, answer, timestamp FROM interactions ORDER BY timestamp DESC"
            ).fetchall()
            
            history = [{
                "id": row['id'],
                "question": row['question'],
                "answer": row['answer'],
                "timestamp": row['timestamp'].split('.')[0] # Keep split for display consistency
            } for row in rows]

            return jsonify(history)
        except Exception as e:
            app.logger.error(f"Error fetching history: {e}", exc_info=True)
            return jsonify({"error": f"Error fetching history: {str(e)}"}), 500

    # --- NEW: Delete Endpoint ---
    @app.route('/history/delete', methods=['POST', 'OPTIONS'])
    def delete_interaction():
        """Deletes a single interaction from the database based on its ID."""
        try:
            data = request.get_json()
            item_id = data.get('id')
            
            if not item_id:
                return jsonify({"error": "Missing 'id' in request body"}), 400

            conn = get_db_connection()
            
            # Use the primary key (id) for a precise and safe delete
            cursor = conn.execute(
                "DELETE FROM interactions WHERE id = ?",
                (item_id,)
            )
            conn.commit()

            if cursor.rowcount == 0:
                app.logger.warning(f"No interaction found to delete with id: {item_id}")
            
            app.logger.info(f"Deleted interaction with id: {item_id}")
            return jsonify({"success": True, "deleted_rows": cursor.rowcount}), 200

        except Exception as e:
            app.logger.error(f"Error deleting history: {e}", exc_info=True)
            return jsonify({"error": f"Error deleting history: {str(e)}"}), 500

    return app

# --- Server Execution ---
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)