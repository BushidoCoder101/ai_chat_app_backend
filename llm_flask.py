import os
import sqlite3
import time
# NEW: Added 'g' to imports for database request context
from flask import Flask, request, jsonify, g
from flask_cors import CORS, cross_origin

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# -------------------------

# --- Configuration ---
class Config:
    """Application configuration settings."""
    
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b") # Changed to a common default
    DATABASE_PATH = os.getenv("DATABASE_PATH", "ollama_chat_v2.db")
    LOG_LEVEL = 'INFO' 
    LLM_SYSTEM_PROMPT = (
        "You are an expert-level personal assistant, skilled in all subjects. "
        "You will be given the current conversation history. "
        "Your responses must be engaging, informative, complete, and formatted in Markdown."
    )

# --- Database Management ---
def get_db_connection():
    """Establishes a connection to the SQLite database and uses 'g' for reuse."""
    # 'g' is used to store data during the lifetime of a single request.
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
    
    # Initialize CORS to allow requests from your frontend.
    # NOTE: using '*' with credentials is disallowed by browsers. If your frontend
    # uses cookies or Authorization headers with `fetch(..., credentials: 'include')`
    # set `supports_credentials=True` and list explicit origins below.
    # Assumption: frontend runs on http://localhost:3000 — change if different.
    CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}}, supports_credentials=True)
    
    init_db(app)
    app.teardown_appcontext(close_db)
    
    app.llm_chain = initialize_llm_chain(app)

    @app.route('/status', methods=['GET'])
    def get_status():
        """Checks the readiness of the LLM chain."""
        if app.llm_chain:
            return jsonify({
                "status": "ready", 
                "model": Config.OLLAMA_MODEL,
                "prompt": Config.LLM_SYSTEM_PROMPT
            }), 200
        else:
            return jsonify({"status": "error", "message": "LLM chain failed to initialize."}), 503

    @app.route('/ask', methods=['POST', 'OPTIONS'])
    @cross_origin()
    def ask_llm():
        """Handles new questions, incorporating chat history."""
        
        # --- Handle OPTIONS preflight request (kept for clarity) ---
        # `@cross_origin` will ensure the proper CORS headers are attached.
        if request.method == 'OPTIONS':
            return jsonify({"success": True}), 200

        if not app.llm_chain:
            return jsonify({"error": f"LLM not ready."}), 503

        try:
            data = request.get_json()
            user_question = data.get('question')
            chat_history = data.get('history', [])
            
            if not user_question:
                return jsonify({"error": "Missing 'question' in request body"}), 400

            app.logger.info(f"Received question: {user_question}")

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

            # Save to database (only if it's the first message of a session)
            if not formatted_history:
                try:
                    conn = get_db_connection()
                    conn.execute(
                        "INSERT INTO interactions (question, answer) VALUES (?, ?)",
                        (user_question, llm_response)
                    )
                    conn.commit()
                except Exception as db_error:
                    app.logger.error(f"Database save error: {db_error}", exc_info=True)


            app.logger.info(f"LLM Answer generated in {end_time - start_time:.2f}s")
            return jsonify({"answer": llm_response})

        except Exception as e:
            app.logger.error(f"Error during chain invocation: {e}", exc_info=True)
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @app.route('/history', methods=['GET'])
    def get_history():
        """Retrieves all past Q&A interactions, newest first."""
        try:
            conn = get_db_connection()
            rows = conn.execute(
                "SELECT id, question, answer, timestamp FROM interactions ORDER BY timestamp DESC"
            ).fetchall()
            
            history = [{
                "id": row['id'],
                "question": row['question'],
                "answer": row['answer'],
                "timestamp": row['timestamp'].split('.')[0]
            } for row in rows]

            return jsonify(history)
        except Exception as e:
            app.logger.error(f"Error fetching history: {e}", exc_info=True)
            return jsonify({"error": f"Error fetching history: {str(e)}"}), 500

    # --- UPDATED: Delete Endpoint ---
    @app.route('/history/delete', methods=['POST', 'OPTIONS'])
    @cross_origin()
    def delete_interaction():
        """Deletes a single interaction from the database based on its ID."""
        
        # --- NEW: Handle OPTIONS preflight request ---
        # This is crucial for CORS when the frontend sends a POST/DELETE
        if request.method == 'OPTIONS':
            # The CORS(app) setup will attach the appropriate headers.
            return jsonify({"success": True}), 200
        # ---------------------------------------------

        # --- POST Logic (moved from try block) ---
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON body provided"}), 400
                
            item_id = data.get('id')
            
            if not item_id:
                return jsonify({"error": "Missing 'id' in request body"}), 400

            conn = get_db_connection()
            
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
