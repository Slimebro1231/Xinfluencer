"""Human evaluation interface for blind content evaluation."""

import logging
import json
import sqlite3
import secrets
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import uuid

logger = logging.getLogger(__name__)


@dataclass
class EvaluationTask:
    """Evaluation task for human reviewers."""
    task_id: str
    prompt: str
    response_a: str
    response_b: str
    response_a_metadata: Dict[str, Any]
    response_b_metadata: Dict[str, Any]
    created_at: datetime
    evaluator_id: Optional[str] = None
    completed_at: Optional[datetime] = None
    preference: Optional[str] = None  # 'A', 'B', or 'Tie'
    confidence: Optional[int] = None  # 1-5 scale
    quality_scores: Optional[Dict[str, int]] = None
    feedback: Optional[str] = None
    evaluation_time_seconds: Optional[int] = None


@dataclass
class EvaluatorProfile:
    """Human evaluator profile."""
    evaluator_id: str
    name: str
    email: str
    expertise_level: str  # 'beginner', 'intermediate', 'expert'
    specializations: List[str]  # ['crypto', 'finance', 'technical_writing', etc.]
    created_at: datetime
    total_evaluations: int = 0
    agreement_rate: Optional[float] = None
    avg_evaluation_time: Optional[float] = None
    is_active: bool = True


class HumanEvaluationDB:
    """Database management for human evaluation system."""
    
    def __init__(self, db_path: str = "data/evaluation/human_eval.db"):
        """Initialize evaluation database."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = self._init_database()
        self.lock = threading.Lock()
    
    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Evaluation tasks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_tasks (
                task_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                response_a TEXT NOT NULL,
                response_b TEXT NOT NULL,
                response_a_metadata TEXT,
                response_b_metadata TEXT,
                created_at TEXT NOT NULL,
                evaluator_id TEXT,
                completed_at TEXT,
                preference TEXT,
                confidence INTEGER,
                quality_scores TEXT,
                feedback TEXT,
                evaluation_time_seconds INTEGER,
                FOREIGN KEY (evaluator_id) REFERENCES evaluators (evaluator_id)
            )
        """)
        
        # Evaluators table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluators (
                evaluator_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                expertise_level TEXT NOT NULL,
                specializations TEXT NOT NULL,
                created_at TEXT NOT NULL,
                total_evaluations INTEGER DEFAULT 0,
                agreement_rate REAL,
                avg_evaluation_time REAL,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Evaluation sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_sessions (
                session_id TEXT PRIMARY KEY,
                evaluator_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                tasks_completed INTEGER DEFAULT 0,
                total_time_seconds INTEGER DEFAULT 0,
                FOREIGN KEY (evaluator_id) REFERENCES evaluators (evaluator_id)
            )
        """)
        
        # Agreement tracking table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agreement_tracking (
                task_id TEXT NOT NULL,
                evaluator_a TEXT NOT NULL,
                evaluator_b TEXT NOT NULL,
                agreement_type TEXT NOT NULL,
                agreement_score REAL NOT NULL,
                calculated_at TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES evaluation_tasks (task_id),
                FOREIGN KEY (evaluator_a) REFERENCES evaluators (evaluator_id),
                FOREIGN KEY (evaluator_b) REFERENCES evaluators (evaluator_id)
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_evaluator ON evaluation_tasks(evaluator_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_completed ON evaluation_tasks(completed_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluators_active ON evaluators(is_active)")
        
        conn.commit()
        return conn
    
    def create_evaluation_task(self, prompt: str, response_a: str, response_b: str,
                             response_a_metadata: Dict = None,
                             response_b_metadata: Dict = None) -> str:
        """Create a new evaluation task."""
        task_id = str(uuid.uuid4())
        
        with self.lock:
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO evaluation_tasks 
                (task_id, prompt, response_a, response_b, response_a_metadata, 
                 response_b_metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id, prompt, response_a, response_b,
                json.dumps(response_a_metadata or {}),
                json.dumps(response_b_metadata or {}),
                datetime.utcnow().isoformat()
            ))
            self.db.commit()
        
        logger.info(f"Created evaluation task: {task_id}")
        return task_id
    
    def get_pending_task(self, evaluator_id: str) -> Optional[EvaluationTask]:
        """Get a pending evaluation task for an evaluator."""
        with self.lock:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT * FROM evaluation_tasks 
                WHERE evaluator_id IS NULL OR evaluator_id = ?
                ORDER BY created_at ASC 
                LIMIT 1
            """, (evaluator_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Assign task to evaluator if not already assigned
            if not row[7]:  # evaluator_id column
                cursor.execute("""
                    UPDATE evaluation_tasks 
                    SET evaluator_id = ? 
                    WHERE task_id = ?
                """, (evaluator_id, row[0]))
                self.db.commit()
            
            return EvaluationTask(
                task_id=row[0],
                prompt=row[1],
                response_a=row[2],
                response_b=row[3],
                response_a_metadata=json.loads(row[4]) if row[4] else {},
                response_b_metadata=json.loads(row[5]) if row[5] else {},
                created_at=datetime.fromisoformat(row[6]),
                evaluator_id=row[7],
                completed_at=datetime.fromisoformat(row[8]) if row[8] else None,
                preference=row[9],
                confidence=row[10],
                quality_scores=json.loads(row[11]) if row[11] else None,
                feedback=row[12],
                evaluation_time_seconds=row[13]
            )
    
    def submit_evaluation(self, task_id: str, evaluator_id: str,
                         preference: str, confidence: int,
                         quality_scores: Dict[str, int],
                         feedback: str = None,
                         evaluation_time_seconds: int = None) -> bool:
        """Submit evaluation results."""
        try:
            with self.lock:
                cursor = self.db.cursor()
                cursor.execute("""
                    UPDATE evaluation_tasks 
                    SET completed_at = ?, preference = ?, confidence = ?,
                        quality_scores = ?, feedback = ?, evaluation_time_seconds = ?
                    WHERE task_id = ? AND evaluator_id = ?
                """, (
                    datetime.utcnow().isoformat(),
                    preference,
                    confidence,
                    json.dumps(quality_scores),
                    feedback,
                    evaluation_time_seconds,
                    task_id,
                    evaluator_id
                ))
                
                # Update evaluator statistics
                cursor.execute("""
                    UPDATE evaluators 
                    SET total_evaluations = total_evaluations + 1
                    WHERE evaluator_id = ?
                """, (evaluator_id,))
                
                self.db.commit()
            
            logger.info(f"Evaluation submitted: {task_id} by {evaluator_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting evaluation: {e}")
            return False
    
    def create_evaluator(self, name: str, email: str, expertise_level: str,
                        specializations: List[str]) -> str:
        """Create a new evaluator profile."""
        evaluator_id = str(uuid.uuid4())
        
        with self.lock:
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO evaluators 
                (evaluator_id, name, email, expertise_level, specializations, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                evaluator_id, name, email, expertise_level,
                json.dumps(specializations),
                datetime.utcnow().isoformat()
            ))
            self.db.commit()
        
        logger.info(f"Created evaluator: {evaluator_id} ({name})")
        return evaluator_id
    
    def get_evaluator(self, evaluator_id: str) -> Optional[EvaluatorProfile]:
        """Get evaluator profile."""
        with self.lock:
            cursor = self.db.cursor()
            cursor.execute("SELECT * FROM evaluators WHERE evaluator_id = ?", (evaluator_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return EvaluatorProfile(
                evaluator_id=row[0],
                name=row[1],
                email=row[2],
                expertise_level=row[3],
                specializations=json.loads(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                total_evaluations=row[6],
                agreement_rate=row[7],
                avg_evaluation_time=row[8],
                is_active=bool(row[9])
            )
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        with self.lock:
            cursor = self.db.cursor()
            
            # Total tasks
            cursor.execute("SELECT COUNT(*) FROM evaluation_tasks")
            total_tasks = cursor.fetchone()[0]
            
            # Completed tasks
            cursor.execute("SELECT COUNT(*) FROM evaluation_tasks WHERE completed_at IS NOT NULL")
            completed_tasks = cursor.fetchone()[0]
            
            # Active evaluators
            cursor.execute("SELECT COUNT(*) FROM evaluators WHERE is_active = 1")
            active_evaluators = cursor.fetchone()[0]
            
            # Recent activity (last 24 hours)
            yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM evaluation_tasks 
                WHERE completed_at >= ?
            """, (yesterday,))
            recent_evaluations = cursor.fetchone()[0]
            
            return {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "pending_tasks": total_tasks - completed_tasks,
                "active_evaluators": active_evaluators,
                "recent_evaluations_24h": recent_evaluations,
                "completion_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            }


class HumanEvaluationApp:
    """Flask web application for human evaluation."""
    
    def __init__(self, db_path: str = "data/evaluation/human_eval.db"):
        """Initialize Flask app."""
        self.app = Flask(__name__, template_folder="templates", static_folder="static")
        self.app.secret_key = secrets.token_hex(32)
        self.db = HumanEvaluationDB(db_path)
        
        # Configure routes
        self._setup_routes()
        
        logger.info("Human evaluation app initialized")
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            if 'evaluator_id' not in session:
                return redirect(url_for('register'))
            
            stats = self.db.get_evaluation_stats()
            evaluator = self.db.get_evaluator(session['evaluator_id'])
            
            return render_template('index.html', stats=stats, evaluator=evaluator)
        
        @self.app.route('/register', methods=['GET', 'POST'])
        def register():
            """Evaluator registration."""
            if request.method == 'POST':
                data = request.json if request.is_json else request.form
                
                evaluator_id = self.db.create_evaluator(
                    name=data['name'],
                    email=data['email'],
                    expertise_level=data['expertise_level'],
                    specializations=data.getlist('specializations') if hasattr(data, 'getlist') else data['specializations']
                )
                
                session['evaluator_id'] = evaluator_id
                
                if request.is_json:
                    return jsonify({'success': True, 'evaluator_id': evaluator_id})
                else:
                    return redirect(url_for('index'))
            
            return render_template('register.html')
        
        @self.app.route('/evaluate')
        def evaluate():
            """Evaluation interface."""
            if 'evaluator_id' not in session:
                return redirect(url_for('register'))
            
            task = self.db.get_pending_task(session['evaluator_id'])
            if not task:
                return render_template('no_tasks.html')
            
            # Randomize response order for blind evaluation
            import random
            if random.choice([True, False]):
                response_order = {'first': 'A', 'second': 'B'}
            else:
                response_order = {'first': 'B', 'second': 'A'}
            
            session['current_task'] = task.task_id
            session['response_order'] = response_order
            session['evaluation_start_time'] = datetime.utcnow().isoformat()
            
            return render_template('evaluate.html', task=task, response_order=response_order)
        
        @self.app.route('/submit_evaluation', methods=['POST'])
        def submit_evaluation():
            """Submit evaluation results."""
            if 'evaluator_id' not in session or 'current_task' not in session:
                return jsonify({'error': 'Invalid session'}), 400
            
            data = request.json
            
            # Calculate evaluation time
            start_time = datetime.fromisoformat(session['evaluation_start_time'])
            evaluation_time = int((datetime.utcnow() - start_time).total_seconds())
            
            # Map response preference back to A/B based on randomized order
            response_order = session['response_order']
            if data['preference'] == 'first':
                preference = response_order['first']
            elif data['preference'] == 'second':
                preference = response_order['second']
            else:
                preference = 'Tie'
            
            success = self.db.submit_evaluation(
                task_id=session['current_task'],
                evaluator_id=session['evaluator_id'],
                preference=preference,
                confidence=data['confidence'],
                quality_scores=data['quality_scores'],
                feedback=data.get('feedback', ''),
                evaluation_time_seconds=evaluation_time
            )
            
            # Clear session data
            session.pop('current_task', None)
            session.pop('response_order', None)
            session.pop('evaluation_start_time', None)
            
            return jsonify({'success': success})
        
        @self.app.route('/stats')
        def stats():
            """Evaluation statistics."""
            if 'evaluator_id' not in session:
                return redirect(url_for('register'))
            
            stats = self.db.get_evaluation_stats()
            evaluator = self.db.get_evaluator(session['evaluator_id'])
            
            return render_template('stats.html', stats=stats, evaluator=evaluator)
        
        @self.app.route('/api/stats')
        def api_stats():
            """API endpoint for statistics."""
            return jsonify(self.db.get_evaluation_stats())
        
        @self.app.route('/logout')
        def logout():
            """Logout evaluator."""
            session.clear()
            return redirect(url_for('register'))
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask app."""
        logger.info(f"Starting human evaluation app on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug) 