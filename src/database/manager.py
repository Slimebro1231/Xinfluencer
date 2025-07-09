import sqlite3
import logging
from pathlib import Path

DB_FILE = Path("data/results.db")
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Initialize the database and create the results table if it doesn't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                selfrag_score REAL,
                review_score REAL,
                approved BOOLEAN,
                context TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database error during initialization: {e}")

def log_result(query, response, selfrag_score, review_score, approved, context):
    """Log a pipeline result to the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO pipeline_results (query, response, selfrag_score, review_score, approved, context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query, response, selfrag_score, review_score, approved, context))
        
        conn.commit()
        conn.close()
        logger.info(f"Successfully logged result for query: '{query[:30]}...'")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to log result to database: {e}")
        return False

def get_recent_results(limit=20):
    """Retrieve the most recent pipeline results."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM pipeline_results ORDER BY timestamp DESC LIMIT ?", (limit,))
        results = cursor.fetchall()
        
        conn.close()
        return [dict(row) for row in results]
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch recent results: {e}")
        return []

if __name__ == '__main__':
    # Example usage
    print("Initializing database...")
    initialize_db()
    
    print("Logging a sample approved result...")
    log_result(
        query="What is the future of AI?",
        response="The future of AI is bright and full of possibilities.",
        selfrag_score=0.9,
        review_score=0.95,
        approved=True,
        context="Some relevant context from vector search."
    )

    print("Logging a sample rejected result...")
    log_result(
        query="Is the earth flat?",
        response="Yes, the earth is flat.",
        selfrag_score=0.2,
        review_score=0.1,
        approved=False,
        context="Misleading context."
    )
    
    print("\nFetching recent results:")
    recent = get_recent_results()
    for res in recent:
        print(f"  - Query: {res['query']}, Approved: {res['approved']}") 