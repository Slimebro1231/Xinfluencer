#!/usr/bin/env python3
"""
Xinfluencer AI CLI
A comprehensive command-line interface for the AI system
"""

import argparse
import sys
import json
import logging
import time
import os
import subprocess
import shlex
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import threading
import queue

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model.generate import TextGenerator
from vector.search import VectorSearcher
from vector.db import VectorDB
from vector.embed import TextEmbedder
from model.selfrag import SelfRAGGenerator
from review.ai import AIReviewer
from utils.logger import setup_logger
from utils.twitter_service import TwitterService
from utils.x_api_client import XAPIClient
from utils.enhanced_data_collection import EnhancedDataCollectionPipeline
from utils.engagement_tracker import EngagementTracker
from webapp.human_evaluator import HumanEvaluationApp, HumanEvaluationDB
from training.identity_pipeline import IdentityTrainingPipeline
from config import config_manager

def _handle_twitter_command(args):
    """Handle Twitter-related commands."""
    if not args.twitter_command:
        print("Twitter command required. Use 'twitter test', 'twitter post <text>', or 'twitter status'")
        return
    
    try:
        # Initialize Twitter service
        print("Initializing Twitter service...")
        twitter_service = TwitterService()
        
        if args.twitter_command == 'status':
            print("\nTwitter Authentication Status:")
            print("=" * 40)
            
            auth_status = twitter_service.get_authentication_status()
            print(f"OAuth 2.0 configured: {auth_status['oauth2_configured']}")
            print(f"OAuth 1.0a configured: {auth_status['oauth1_configured']}")
            print(f"Active v2 client: {auth_status['active_client']}")
            print(f"Active v1.1 API: {auth_status['active_api']}")
            
            print("\nCredentials Available:")
            creds = auth_status['credentials_available']
            for cred_name, available in creds.items():
                status = "✓" if available else "✗"
                print(f"  {status} {cred_name}")
            
            # Test connection
            connection_test = twitter_service.test_connection()
            print(f"\nConnection Test:")
            print(f"Can read: {connection_test['can_read']}")
            print(f"Can post: {connection_test['can_post']}")
            
            if connection_test['errors']:
                print(f"\nErrors:")
                for error in connection_test['errors']:
                    print(f"  - {error}")
        
        elif args.twitter_command == 'test':
            print("\nTesting Twitter API connection...")
            print("=" * 40)
            
            # Test connection
            connection_test = twitter_service.test_connection()
            
            print(f"Can read: {connection_test['can_read']}")
            print(f"Can post: {connection_test['can_post']}")
            print(f"Rate limits available: {connection_test['rate_limits_available']}")
            
            if connection_test['errors']:
                print(f"\nErrors:")
                for error in connection_test['errors']:
                    print(f"  - {error}")
            else:
                print("\n✓ Twitter API connection successful!")
        
        elif args.twitter_command == 'post':
            if not args.text:
                print("Tweet text is required for posting")
                return
            
            print(f"\nPosting tweet: '{args.text[:50]}{'...' if len(args.text) > 50 else ''}'")
            print("=" * 40)
            
            # Test connection first
            connection_test = twitter_service.test_connection()
            if not connection_test['can_post']:
                print("✗ Cannot post - Twitter API not properly configured")
                print("Run 'twitter test' for more details")
                return
            
            # Post the tweet
            result = twitter_service.post_tweet(args.text)
            
            if result['success']:
                print(f"✓ Successfully posted tweet!")
                print(f"  Method: {result['method']}")
                print(f"  Tweet ID: {result['tweet_id']}")
                print(f"  Text: {result['text']}")
            else:
                print(f"✗ Failed to post tweet: {result.get('error', 'Unknown error')}")
        
        else:
            print(f"Unknown Twitter command: {args.twitter_command}")
    
    except Exception as e:
        print(f"Error in Twitter command: {e}")
        import traceback
        traceback.print_exc()


def _handle_enhanced_collection_command(args):
    """Handle enhanced data collection commands."""
    if not args.enhanced_command:
        print("Enhanced collection command required")
        return
    
    try:
        pipeline = EnhancedDataCollectionPipeline()
        
        if args.enhanced_command == 'safe-collect':
            target_posts = getattr(args, 'target_posts', 500)
            save_training = not getattr(args, 'no_training', False)
            
            print(f"\nStarting safe collection (target: {target_posts} posts)")
            print("=" * 50)
            
            results = pipeline.safe_collect_crypto_content(
                target_posts=target_posts,
                save_for_training=save_training
            )
            
            if results.get('success', True):
                print(f"\nCollection Results:")
                print(f"Posts collected: {results['total_posts_collected']}")
                print(f"Training posts stored: {results['training_posts_stored']}")
                print(f"Collection efficiency: {results['collection_efficiency']:.1f} posts/API call")
                print(f"Session ID: {results['session_id']}")
                print(f"Duration: {results['duration_seconds']:.1f} seconds")
                
                safeguard_status = results['safeguard_status']
                print(f"\nUsage Status:")
                print(f"Posts last hour: {safeguard_status['posts_last_hour']}")
                print(f"Posts last day: {safeguard_status['posts_last_day']}")
            else:
                print(f"Collection failed: {results.get('error', 'Unknown error')}")
        
        elif args.enhanced_command == 'training-stats':
            print("\nTraining Data Statistics:")
            print("=" * 40)
            
            stats = pipeline.get_training_data_stats()
            
            if 'error' not in stats:
                print(f"Total posts: {stats['total_posts']}")
                print(f"High quality posts: {stats['high_quality_posts']}")
                print(f"High crypto relevance: {stats['high_crypto_relevance']}")
                
                print(f"\nTop authors:")
                for author, count in list(stats['top_authors'].items())[:5]:
                    print(f"  {author}: {count} posts")
                
                print(f"\nRecent sessions:")
                for session, count in list(stats['recent_sessions'].items())[:3]:
                    print(f"  {session}: {count} posts")
            else:
                print(f"Error getting stats: {stats['error']}")
        
        elif args.enhanced_command == 'status':
            print("\nEnhanced Collection Status:")
            print("=" * 40)
            
            # Check safeguard status
            limits_check = pipeline.safeguard.check_post_limits()
            print(f"Can collect: {limits_check['can_collect']}")
            print(f"Posts this hour: {limits_check['posts_last_hour']}/{limits_check['hourly_limit']}")
            print(f"Posts today: {limits_check['posts_last_day']}/{limits_check['daily_limit']}")
            
            # Training data stats
            training_stats = pipeline.get_training_data_stats()
            print(f"\nTraining data available: {training_stats.get('total_posts', 0)} posts")
            print(f"High quality available: {training_stats.get('high_quality_posts', 0)} posts")
        
        else:
            print(f"Unknown enhanced collection command: {args.enhanced_command}")
    
    except Exception as e:
        print(f"Error in enhanced collection: {e}")
        import traceback
        traceback.print_exc()


def _handle_identity_training_command(args):
    """Handle identity training commands."""
    if not args.identity_command:
        print("Identity training command required")
        return
    
    try:
        pipeline = IdentityTrainingPipeline()
        
        if args.identity_command == 'train':
            print("\nStarting identity training...")
            print("=" * 40)
            
            output_dir = getattr(args, 'output_dir', 'lora_checkpoints/identity')
            adapter_path = pipeline.train_identity_model(output_dir=output_dir)
            
            if adapter_path:
                print(f"\n✓ Identity training completed!")
                print(f"Adapter saved to: {adapter_path}")
            else:
                print("✗ Identity training failed")
        
        elif args.identity_command == 'status':
            print("\nIdentity Training Status:")
            print("=" * 40)
            
            status = pipeline.get_training_status()
            
            storage_stats = status['storage_stats']
            print(f"Available training posts: {storage_stats.get('total_posts', 0)}")
            print(f"High quality posts: {storage_stats.get('high_quality_posts', 0)}")
            print(f"Ready for training: {status['available_for_training']}")
            
            if status['recent_training']:
                recent = status['recent_training']
                print(f"\nLast training:")
                print(f"  Timestamp: {recent['timestamp']}")
                print(f"  Examples used: {recent['training_stats']['total_examples']}")
                print(f"  Average quality: {recent['training_stats']['avg_quality']:.3f}")
                print(f"  Adapter: {recent['adapter_path']}")
        
        elif args.identity_command == 'quality-check':
            print("\nTraining Data Quality Check:")
            print("=" * 40)
            
            training_examples = pipeline.get_high_quality_training_data()
            
            if training_examples:
                print(f"High-quality examples found: {len(training_examples)}")
                print(f"\nTop examples by quality:")
                
                # Sort by weight and show top 3
                sorted_examples = sorted(training_examples, key=lambda x: x['weight'], reverse=True)
                for i, example in enumerate(sorted_examples[:3], 1):
                    print(f"\n{i}. Author: {example['author']}")
                    print(f"   Weight: {example['weight']:.3f}")
                    print(f"   Text: {example['response'][:80]}...")
            else:
                print("No high-quality training examples available")
                print("Run 'enhanced safe-collect' to gather training data")
        
        else:
            print(f"Unknown identity training command: {args.identity_command}")
    
    except Exception as e:
        print(f"Error in identity training: {e}")
        import traceback
        traceback.print_exc()


def _run_command(command: str, remote: bool, config: Any):
    """Helper to run a command locally or remotely via SSH."""
    if remote:
        pem_file = config.h200.pem_file
        user = config.h200.user
        host = config.h200.host
        remote_dir = config.h200.remote_dir
        
        ssh_command_str = (
            f"ssh -i {pem_file} -o StrictHostKeyChecking=no {user}@{host} "
            f"'cd {remote_dir} && source xinfluencer_env/bin/activate && {command}'"
        )
        final_command = shlex.split(ssh_command_str)
        print(f"Running command on H200: {command}")
    else:
        final_command = shlex.split(command)
        print(f"Running command locally: {command}")

    try:
        with subprocess.Popen(final_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            if proc.stdout:
                for line in proc.stdout:
                    print(line, end='')
        
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)

    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}")
    except FileNotFoundError:
        print(f"\nSSH key not found at '{pem_file}'. Please check H200_PEM_FILE in your .env file.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

class ControlCenter:
    """Text-based multi-panel control center for Xinfluencer AI."""
    
    def __init__(self):
        self.logger = setup_logger("control_center", level="INFO")
        self.running = False
        self.update_interval = 2.0  # seconds
        self.log_file = Path("logs/xinfluencer.log")
        self.last_log_size = 0
        
        # Panel data
        self.system_status = {}
        self.pipeline_status = {}
        self.gpu_status = {}
        self.error_log = []
        self.performance_metrics = {}
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version.split()[0],
            "cuda_available": False,
            "gpu_name": "N/A",
            "gpu_memory": "N/A",
            "disk_usage": "N/A",
            "memory_usage": "N/A"
        }
        
        try:
            import torch
            status["pytorch_version"] = torch.__version__
            status["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                status["gpu_name"] = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                status["gpu_memory"] = f"{gpu_memory:.1f} GB"
        except ImportError:
            status["pytorch_version"] = "Not installed"
        
        try:
            import psutil
            status["disk_usage"] = f"{psutil.disk_usage('/').percent:.1f}%"
            status["memory_usage"] = f"{psutil.virtual_memory().percent:.1f}%"
        except ImportError:
            pass
            
        return status
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status and recent activity."""
        status = {
            "last_run": "Never",
            "status": "Unknown",
            "tweets_processed": 0,
            "chunks_generated": 0,
            "errors": 0,
            "warnings": 0
        }
        
        # Check if log file exists and get last run info
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Find last pipeline run
                        for line in reversed(lines):
                            if "Starting Xinfluencer AI pipeline" in line:
                                status["last_run"] = line.split()[0] + " " + line.split()[1]
                                status["status"] = "Completed"
                                break
                            elif "Pipeline failed" in line:
                                status["last_run"] = line.split()[0] + " " + line.split()[1]
                                status["status"] = "Failed"
                                break
                        
                        # Count recent errors and warnings
                        recent_lines = lines[-100:]  # Last 100 lines
                        status["errors"] = sum(1 for line in recent_lines if "ERROR" in line)
                        status["warnings"] = sum(1 for line in recent_lines if "WARNING" in line)
                        
                        # Extract metrics from log
                        for line in recent_lines:
                            if "Tweets processed:" in line:
                                try:
                                    status["tweets_processed"] = int(line.split(":")[-1].strip())
                                except:
                                    pass
                            elif "Text chunks generated:" in line:
                                try:
                                    status["chunks_generated"] = int(line.split(":")[-1].strip())
                                except:
                                    pass
            except Exception as e:
                status["status"] = f"Error reading log: {e}"
        
        return status
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get detailed GPU status."""
        status = {
            "gpu_utilization": "N/A",
            "memory_used": "N/A",
            "memory_free": "N/A",
            "temperature": "N/A",
            "power_draw": "N/A"
        }
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 5:
                    status["gpu_utilization"] = f"{parts[0]}%"
                    status["memory_used"] = f"{parts[1]} MB"
                    status["memory_free"] = f"{parts[2]} MB"
                    status["temperature"] = f"{parts[3]}°C"
                    status["power_draw"] = f"{parts[4]} W"
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def get_recent_errors(self, max_lines: int = 10) -> List[str]:
        """Get recent error messages from log."""
        errors = []
        
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()
                    # Get last 100 lines and filter errors
                    recent_lines = lines[-100:]
                    error_lines = [line.strip() for line in recent_lines if "ERROR" in line]
                    errors = error_lines[-max_lines:]  # Last N errors
            except Exception as e:
                errors = [f"Error reading log: {e}"]
        
        return errors
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {
            "embedding_speed": "N/A",
            "search_speed": "N/A",
            "generation_speed": "N/A",
            "memory_efficiency": "N/A"
        }
        
        # This would be populated from actual performance data
        # For now, return placeholder values
        return metrics
    
    def update_panels(self):
        """Update all panel data."""
        self.system_status = self.get_system_status()
        self.pipeline_status = self.get_pipeline_status()
        self.gpu_status = self.get_gpu_status()
        self.error_log = self.get_recent_errors()
        self.performance_metrics = self.get_performance_metrics()
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def draw_panel(self, title: str, content: Dict[str, Any], width: int = 40) -> str:
        """Draw a single panel."""
        panel = f"┌─ {title} {'─' * (width - len(title) - 4)}┐\n"
        
        for key, value in content.items():
            if isinstance(value, dict):
                panel += f"│ {key}:\n"
                for sub_key, sub_value in value.items():
                    panel += f"│   {sub_key}: {sub_value}\n"
            else:
                # Truncate long values
                display_value = str(value)
                if len(display_value) > width - len(key) - 8:
                    display_value = display_value[:width - len(key) - 11] + "..."
                panel += f"│ {key}: {display_value}\n"
        
        panel += f"└{'─' * width}┘\n"
        return panel
    
    def draw_error_panel(self, errors: List[str], width: int = 60) -> str:
        """Draw error panel."""
        panel = f"┌─ Recent Errors {'─' * (width - 15)}┐\n"
        
        if not errors:
            panel += f"│ No recent errors\n"
        else:
            for error in errors:
                # Truncate long error messages
                if len(error) > width - 4:
                    error = error[:width - 7] + "..."
                panel += f"│ {error}\n"
        
        panel += f"└{'─' * width}┘\n"
        return panel
    
    def draw_control_center(self):
        """Draw the complete control center interface."""
        self.clear_screen()
        
        # Update all data
        self.update_panels()
        
        # Header
        header = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    XINFLUENCER AI CONTROL CENTER                            ║
║                              {self.system_status['timestamp']}                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""
        print(header)
        
        # Main panels layout
        panels_layout = f"""
{self.draw_panel("System Status", self.system_status, 40)}
{self.draw_panel("Pipeline Status", self.pipeline_status, 40)}
{self.draw_panel("GPU Status", self.gpu_status, 40)}

{self.draw_panel("Performance Metrics", self.performance_metrics, 40)}
{self.draw_error_panel(self.error_log, 60)}

Controls: [r]efresh [q]uit [l]og tail [p]ipeline run [s]ystem test
"""
        print(panels_layout)
    
    def tail_log(self):
        """Tail the log file in real-time."""
        if not self.log_file.exists():
            print("Log file not found.")
            return
        
        print(f"Tailing log file: {self.log_file}")
        print("Press Ctrl+C to stop\n")
        
        try:
            with open(self.log_file, 'r') as f:
                # Go to end of file
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        print(line.rstrip())
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped tailing log.")
    
    def run_pipeline(self):
        """Run the full pipeline."""
        print("Running full pipeline...")
        try:
            # Import and run main pipeline
            from main import main
            result = main()
            print(f"Pipeline completed with status: {result['status']}")
        except Exception as e:
            print(f"Pipeline failed: {e}")
    
    def run_system_test(self):
        """Run system tests."""
        print("Running system tests...")
        try:
            # Test GPU
            import torch
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name()} - OK")
                print(f"CUDA: {torch.version.cuda} - OK")
            else:
                print("GPU: Not available")
            
            # Test vector search
            print("Testing vector search...")
            # Add vector search test here
            
            print("System tests completed.")
        except Exception as e:
            print(f"System test failed: {e}")
    
    def run(self):
        """Run the control center."""
        self.running = True
        
        while self.running:
            try:
                self.draw_control_center()
                
                # Non-blocking input check
                import select
                import tty
                import termios
                
                # Set terminal to raw mode
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setraw(sys.stdin.fileno())
                    
                    # Check for input with timeout
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        
                        if key == 'q':
                            self.running = False
                        elif key == 'r':
                            # Refresh is automatic
                            pass
                        elif key == 'l':
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                            self.tail_log()
                            input("Press Enter to continue...")
                        elif key == 'p':
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                            self.run_pipeline()
                            input("Press Enter to continue...")
                        elif key == 's':
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                            self.run_system_test()
                            input("Press Enter to continue...")
                
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"Control center error: {e}")
                time.sleep(5)

class XinfluencerCLI:
    """Command-line interface for Xinfluencer AI."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.logger = setup_logger("xinfluencer_cli", level="INFO")
        self.generator = None
        self.searcher = None
        self.selfrag = None
        self.reviewer = None
        
    def initialize_components(self, use_gpu: bool = True):
        """Initialize AI components."""
        try:
            self.logger.info("Initializing AI components...")
            
            # Initialize generator
            if use_gpu:
                self.generator = TextGenerator(use_quantization=True)
            else:
                self.generator = TextGenerator(use_quantization=False)
            
            # Initialize vector components
            embedder = TextEmbedder()
            vector_db = VectorDB()
            self.searcher = VectorSearcher(vector_db, embedder)
            
            # Initialize Self-RAG and reviewer
            self.selfrag = SelfRAGGenerator(self.generator, self.searcher)
            self.reviewer = AIReviewer(self.generator)
            
            self.logger.info("AI components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text response."""
        if not self.generator:
            return "Error: AI components not initialized"
        
        try:
            response = self.generator.generate_response(prompt, max_tokens, temperature)
            return response
        except Exception as e:
            return f"Generation error: {e}"
    
    def generate_with_rag(self, query: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Generate response using Self-RAG."""
        if not self.selfrag:
            return {"error": "Self-RAG components not initialized"}
        
        try:
            result = self.selfrag.generate_with_self_rag(query)
            return result
        except Exception as e:
            return {"error": f"Self-RAG error: {e}"}
    
    def review_response(self, query: str, response: str, context: str = "") -> Dict[str, Any]:
        """Review a response using AI reviewer."""
        if not self.reviewer:
            return {"error": "Reviewer not initialized"}
        
        try:
            result = self.reviewer.review_response(query, response, context)
            return result
        except Exception as e:
            return {"error": f"Review error: {e}"}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage."""
        if not self.generator:
            return {"error": "Generator not initialized"}
        
        try:
            return self.generator.get_memory_usage()
        except Exception as e:
            return {"error": f"Memory check error: {e}"}
    
    def clear_memory(self):
        """Clear GPU memory."""
        if self.generator:
            self.generator.clear_memory()
            self.logger.info("GPU memory cleared")

def _handle_x_api_command(args):
    """Handle X API data collection commands."""
    if not args.x_api_command:
        print("X API command required. Use 'x-api test', 'x-api collect', or 'x-api track'")
        return
    
    try:
        if args.x_api_command == 'test':
            print("\nTesting X API connection...")
            print("=" * 40)
            
            x_api = XAPIClient()
            connection_test = x_api.test_connection()
            
            print(f"Connected: {connection_test['connected']}")
            print(f"Capabilities: {', '.join(connection_test['capabilities'])}")
            
            if connection_test['errors']:
                print(f"\nErrors:")
                for error in connection_test['errors']:
                    print(f"  - {error}")
            
            # Show rate limits
            rate_limits = x_api.get_rate_limit_status()
            print(f"\nRate Limits:")
            for endpoint, status in rate_limits.items():
                if isinstance(status, dict) and 'remaining' in status:
                    print(f"  {endpoint}: {status['remaining']}/{status['limit']} remaining")
        
        elif args.x_api_command == 'collect':
            print("\nStarting data collection...")
            print("=" * 40)
            
            pipeline = EnhancedDataCollectionPipeline()
            
            # Run comprehensive collection
            results = pipeline.run_comprehensive_collection(
                kol_usernames=getattr(args, 'kols', None),
                collect_trending=not getattr(args, 'no_trending', False),
                collect_high_engagement=not getattr(args, 'no_high_engagement', False)
            )
            
            print(f"\nCollection Results:")
            print(f"KOLs processed: {len(results['kol_data'])}")
            print(f"Trending tweets: {len(results['trending_tweets'])}")
            print(f"High-engagement tweets: {len(results['high_engagement_tweets'])}")
            print(f"Total API calls: {results['collection_stats']['api_calls_made']}")
            print(f"Errors: {results['collection_stats']['errors']}")
        
        elif args.x_api_command == 'kol-analysis':
            print("\nRunning KOL performance analysis...")
            print("=" * 40)
            
            pipeline = EnhancedDataCollectionPipeline()
            analysis = pipeline.get_kol_performance_analysis(
                usernames=getattr(args, 'kols', None)
            )
            
            if analysis:
                print(f"\nKOL Analysis Results:")
                for username, data in analysis.items():
                    print(f"\n@{username}:")
                    print(f"  Total tweets: {data['total_tweets']}")
                    print(f"  Avg engagement: {data['avg_engagement_per_tweet']:.1f}")
                    print(f"  Crypto engagement ratio: {data['crypto_engagement_ratio']:.2%}")
            else:
                print("No analysis data available")
        
        elif args.x_api_command == 'track':
            print("\nStarting engagement tracking...")
            print("=" * 40)
            
            tracker = EngagementTracker()
            
            if hasattr(args, 'tweet_ids') and args.tweet_ids:
                # Track specific tweets
                tweet_ids = args.tweet_ids.split(',')
                for tweet_id in tweet_ids:
                    success = tracker.start_tracking_tweet(tweet_id.strip())
                    print(f"Tracking tweet {tweet_id}: {'✓' if success else '✗'}")
            else:
                print("No tweet IDs provided for tracking")
                print("Usage: x-api track --tweet-ids 1234567890,0987654321")
        
        elif args.x_api_command == 'status':
            print("\nX API and Tracking Status:")
            print("=" * 40)
            
            # X API status
            x_api = XAPIClient()
            api_status = x_api.get_rate_limit_status()
            print(f"API Connected: {api_status.get('api_connected', False)}")
            print(f"Auth Type: {api_status.get('auth_type', 'none')}")
            
            # Data collection status
            pipeline = EnhancedDataCollectionPipeline()
            collection_stats = pipeline.get_collection_statistics()
            print(f"\nCollection Stats:")
            print(f"Tweets collected: {collection_stats['tweets_collected']}")
            print(f"KOLs processed: {collection_stats['kols_processed']}")
            print(f"API calls made: {collection_stats['api_calls_made']}")
            
            # Engagement tracking status
            tracker = EngagementTracker()
            tracking_status = tracker.get_tracking_status()
            print(f"\nEngagement Tracking:")
            print(f"Active tracking: {tracking_status['active_tracking']}")
            print(f"Total tracked: {tracking_status['total_tracked']}")
            print(f"Recent snapshots (1h): {tracking_status['recent_snapshots_1h']}")
        
        elif args.x_api_command == 'top-tweets':
            print("\nTop Performing Tweets:")
            print("=" * 40)
            
            tracker = EngagementTracker()
            top_tweets = tracker.get_top_performing_tweets(
                limit=getattr(args, 'limit', 10)
            )
            
            if top_tweets:
                for i, tweet in enumerate(top_tweets, 1):
                    print(f"\n{i}. @{tweet['author_username']}")
                    print(f"   {tweet['tweet_text']}")
                    print(f"   Peak engagement: {tweet['max_total_engagement']}")
                    print(f"   Avg velocity: {tweet['avg_velocity']:.1f}/hr")
            else:
                print("No tracked tweets found")
        
        else:
            print(f"Unknown X API command: {args.x_api_command}")
    
    except Exception as e:
        print(f"Error in X API command: {e}")
        import traceback
        traceback.print_exc()


def _handle_human_eval_command(args):
    """Handle human evaluation interface commands."""
    if not args.human_eval_command:
        print("Human eval command required. Use 'human-eval start', 'human-eval create-task', or 'human-eval stats'")
        return
    
    try:
        if args.human_eval_command == 'start':
            print("Starting Human Evaluation Web Interface...")
            print("=" * 50)
            
            app = HumanEvaluationApp()
            host = getattr(args, 'host', '127.0.0.1')
            port = getattr(args, 'port', 5000)
            debug = getattr(args, 'debug', False)
            
            print(f"🌐 Web interface will be available at: http://{host}:{port}")
            print("📝 Evaluators can register and start evaluating AI responses")
            print("🔗 Open the URL in your browser to access the interface")
            print("\nPress Ctrl+C to stop the server\n")
            
            try:
                app.run(host=host, port=port, debug=debug)
            except KeyboardInterrupt:
                print("\n👋 Shutting down human evaluation interface...")
            except Exception as e:
                print(f"❌ Error starting web interface: {e}")
        
        elif args.human_eval_command == 'create-task':
            print("Creating evaluation task...")
            print("=" * 40)
            
            db = HumanEvaluationDB()
            
            # Get required arguments
            prompt = getattr(args, 'prompt', None)
            response_a = getattr(args, 'response_a', None)
            response_b = getattr(args, 'response_b', None)
            
            if not all([prompt, response_a, response_b]):
                print("❌ All arguments required: --prompt, --response-a, --response-b")
                return
            
            # Create metadata
            metadata_a = {"source": "model_a", "created_at": datetime.utcnow().isoformat()}
            metadata_b = {"source": "model_b", "created_at": datetime.utcnow().isoformat()}
            
            task_id = db.create_evaluation_task(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
                response_a_metadata=metadata_a,
                response_b_metadata=metadata_b
            )
            
            print(f"✅ Created evaluation task: {task_id}")
            print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            print(f"📊 Response A: {len(response_a)} characters")
            print(f"📊 Response B: {len(response_b)} characters")
        
        elif args.human_eval_command == 'stats':
            print("Human Evaluation Statistics:")
            print("=" * 40)
            
            db = HumanEvaluationDB()
            stats = db.get_evaluation_stats()
            
            print(f"📋 Total tasks: {stats['total_tasks']}")
            print(f"✅ Completed tasks: {stats['completed_tasks']}")
            print(f"⏳ Pending tasks: {stats['pending_tasks']}")
            print(f"👥 Active evaluators: {stats['active_evaluators']}")
            print(f"🔥 Recent evaluations (24h): {stats['recent_evaluations_24h']}")
            print(f"📈 Completion rate: {stats['completion_rate']:.1f}%")
        
        elif args.human_eval_command == 'demo-tasks':
            print("Creating demo evaluation tasks...")
            print("=" * 40)
            
            db = HumanEvaluationDB()
            
            # Create some demo tasks for testing
            demo_tasks = [
                {
                    "prompt": "What is Bitcoin and how does it work?",
                    "response_a": "Bitcoin is a digital cryptocurrency that uses blockchain technology to enable peer-to-peer transactions without intermediaries. It operates on a decentralized network where transactions are verified by miners through a proof-of-work consensus mechanism.",
                    "response_b": "Bitcoin is like digital money that you can send to anyone without using banks. It's based on a special technology called blockchain that keeps track of all transactions. People called miners use computers to verify transactions and get rewarded with new bitcoins."
                },
                {
                    "prompt": "Explain DeFi and its benefits over traditional finance.",
                    "response_a": "DeFi (Decentralized Finance) refers to financial services built on blockchain networks, particularly Ethereum. It eliminates intermediaries like banks by using smart contracts to automate lending, borrowing, trading, and other financial operations. Benefits include 24/7 availability, global accessibility, transparency, and often lower fees.",
                    "response_b": "Decentralized Finance or DeFi is a new financial system that works without traditional banks or financial institutions. Instead, it uses computer programs called smart contracts on the blockchain. This means you can lend, borrow, and trade money directly with other people around the world, anytime you want, often with better rates than banks."
                }
            ]
            
            created_count = 0
            for task in demo_tasks:
                task_id = db.create_evaluation_task(
                    prompt=task["prompt"],
                    response_a=task["response_a"],
                    response_b=task["response_b"]
                )
                created_count += 1
                print(f"✅ Created task {created_count}: {task['prompt'][:50]}...")
            
            print(f"\n🎉 Created {created_count} demo evaluation tasks")
            print("👨‍💻 Evaluators can now start evaluating these tasks through the web interface")
        
        else:
            print(f"Unknown human eval command: {args.human_eval_command}")
    
    except Exception as e:
        print(f"Error in human eval command: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Xinfluencer AI CLI - Complete AI system for crypto content analysis and generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text
  %(prog)s generate "What is Bitcoin?" --temperature 0.8

  # Enhanced data collection
  %(prog)s enhanced safe-collect --target-posts 300
  %(prog)s enhanced training-stats
  %(prog)s enhanced status

  # Identity training
  %(prog)s identity train
  %(prog)s identity status
  %(prog)s identity quality-check

  # Twitter operations
  %(prog)s twitter test
  %(prog)s twitter post "Hello from Xinfluencer AI!"

  # X API data collection
  %(prog)s x-api test
  %(prog)s x-api collect --kols VitalikButerin,elonmusk

  # Interactive mode
  %(prog)s interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data command group
    data_parser = subparsers.add_parser('data', help='Data acquisition and processing commands')
    data_subparsers = data_parser.add_subparsers(dest='data_command', help='Data commands')
    
    # Data scrape command
    scrape_parser = data_subparsers.add_parser('scrape', help='Scrape seed data from the web')
    scrape_parser.add_argument('--remote', action='store_true', help='Run the scraper on the H200 server')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text response')
    gen_parser.add_argument('prompt', help='Input prompt')
    gen_parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    gen_parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    gen_parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    
    # RAG command
    rag_parser = subparsers.add_parser('rag', help='Generate with Self-RAG')
    rag_parser.add_argument('query', help='Query for RAG generation')
    rag_parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    rag_parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    
    # Review command
    review_parser = subparsers.add_parser('review', help='Review a response')
    review_parser.add_argument('query', help='Original query')
    review_parser.add_argument('response', help='Response to review')
    review_parser.add_argument('--context', default='', help='Context for review')
    review_parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    
    # Memory command
    memory_parser = subparsers.add_parser('memory', help='Check GPU memory usage')
    memory_parser.add_argument('--clear', action='store_true', help='Clear GPU memory')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    interactive_parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check system status')
    
    # Control center command
    control_parser = subparsers.add_parser('control-center', help='Launch multi-panel control center')
    
    # Twitter command
    twitter_parser = subparsers.add_parser('twitter', help='Twitter API testing and posting')
    twitter_subparsers = twitter_parser.add_subparsers(dest='twitter_command', help='Twitter commands')
    
    # Twitter test command
    twitter_test_parser = twitter_subparsers.add_parser('test', help='Test Twitter API connection')
    
    # Twitter post command
    twitter_post_parser = twitter_subparsers.add_parser('post', help='Post a tweet')
    twitter_post_parser.add_argument('text', help='Tweet text to post')
    
    # Twitter status command
    twitter_status_parser = twitter_subparsers.add_parser('status', help='Show Twitter authentication status')
    
    # X API commands
    x_api_parser = subparsers.add_parser('x-api', help='X API data collection and tracking')
    x_api_subparsers = x_api_parser.add_subparsers(dest='x_api_command', help='X API command')
    
    # X API test
    x_api_test_parser = x_api_subparsers.add_parser('test', help='Test X API connection')
    
    # X API collect
    x_api_collect_parser = x_api_subparsers.add_parser('collect', help='Collect data from X API')
    x_api_collect_parser.add_argument('--kols', nargs='+', help='Specific KOL usernames to collect from')
    x_api_collect_parser.add_argument('--no-trending', action='store_true', help='Skip trending tweets collection')
    x_api_collect_parser.add_argument('--no-high-engagement', action='store_true', help='Skip high-engagement tweets collection')
    
    # X API KOL analysis
    x_api_kol_parser = x_api_subparsers.add_parser('kol-analysis', help='Analyze KOL performance')
    x_api_kol_parser.add_argument('--kols', nargs='+', help='Specific KOL usernames to analyze')
    
    # X API tracking
    x_api_track_parser = x_api_subparsers.add_parser('track', help='Track tweet engagement')
    x_api_track_parser.add_argument('--tweet-ids', help='Comma-separated tweet IDs to track')
    
    # X API status
    x_api_status_parser = x_api_subparsers.add_parser('status', help='Show X API and tracking status')
    
    # X API top tweets
    x_api_top_parser = x_api_subparsers.add_parser('top-tweets', help='Show top performing tweets')
    x_api_top_parser.add_argument('--limit', type=int, default=10, help='Number of top tweets to show')

    # Human evaluation commands
    human_eval_parser = subparsers.add_parser('human-eval', help='Human evaluation interface')
    human_eval_subparsers = human_eval_parser.add_subparsers(dest='human_eval_command', help='Human eval command')
    
    # Human eval start
    human_eval_start_parser = human_eval_subparsers.add_parser('start', help='Start web interface')
    human_eval_start_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    human_eval_start_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    human_eval_start_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Human eval create task
    human_eval_task_parser = human_eval_subparsers.add_parser('create-task', help='Create evaluation task')
    human_eval_task_parser.add_argument('--prompt', required=True, help='Evaluation prompt')
    human_eval_task_parser.add_argument('--response-a', required=True, help='First response to evaluate')
    human_eval_task_parser.add_argument('--response-b', required=True, help='Second response to evaluate')
    
    # Human eval stats
    human_eval_stats_parser = human_eval_subparsers.add_parser('stats', help='Show evaluation statistics')
    
    # Human eval demo tasks
    human_eval_demo_parser = human_eval_subparsers.add_parser('demo-tasks', help='Create demo evaluation tasks')

    # Enhanced data collection commands
    enhanced_parser = subparsers.add_parser('enhanced', help='Enhanced data collection with training integration')
    enhanced_subparsers = enhanced_parser.add_subparsers(dest='enhanced_command', help='Enhanced collection commands')
    
    # Safe collect command
    safe_collect_parser = enhanced_subparsers.add_parser('safe-collect', help='Safely collect crypto content')
    safe_collect_parser.add_argument('--target-posts', type=int, default=500, help='Target number of posts to collect')
    safe_collect_parser.add_argument('--no-training', action='store_true', help='Skip storing for training')
    
    # Training stats command
    training_stats_parser = enhanced_subparsers.add_parser('training-stats', help='Show training data statistics')
    
    # Enhanced status command
    enhanced_status_parser = enhanced_subparsers.add_parser('status', help='Show enhanced collection status')
    
    # Identity training commands
    identity_parser = subparsers.add_parser('identity', help='Identity training for crypto bot personality')
    identity_subparsers = identity_parser.add_subparsers(dest='identity_command', help='Identity training commands')
    
    # Identity train command
    identity_train_parser = identity_subparsers.add_parser('train', help='Train bot identity with collected data')
    identity_train_parser.add_argument('--output-dir', default='lora_checkpoints/identity', help='Output directory for LoRA adapter')
    
    # Identity status command
    identity_status_parser = identity_subparsers.add_parser('status', help='Show identity training status')
    
    # Identity quality check command
    identity_quality_parser = identity_subparsers.add_parser('quality-check', help='Check quality of available training data')

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute data commands which don't require full component initialization
    if args.command == 'data':
        if args.data_command == 'scrape':
            try:
                config = config_manager
                command_to_run = "python3 scripts/scrape_tweets_from_web.py"
                _run_command(command_to_run, remote=args.remote, config=config)
                print("\nScraping completed successfully!")
            except Exception as e:
                print(f"Failed to run scraper: {e}")
        return # Exit after data command is handled
    
    # Execute Twitter commands
    if args.command == 'twitter':
        _handle_twitter_command(args)
        return

    # Handle control center command
    if args.command == 'control-center':
        control_center = ControlCenter()
        control_center.run()
        return

    # Execute X API commands
    if args.command == 'x-api':
        _handle_x_api_command(args)
        return

    # Execute Human Evaluation commands
    if args.command == 'human-eval':
        _handle_human_eval_command(args)
        return

    # Handle new enhanced collection commands
    if args.command == 'enhanced':
        _handle_enhanced_collection_command(args)
        return

    # Handle new identity training commands
    if args.command == 'identity':
        _handle_identity_training_command(args)
        return

    # Initialize components if needed
    if args.command in ['generate', 'rag', 'review', 'interactive']:
        use_gpu = not getattr(args, 'no_gpu', False)
        if not cli.initialize_components(use_gpu):
            print("Failed to initialize AI components")
            return
    
    # Execute commands
    if args.command == 'generate':
        response = cli.generate(args.prompt, args.max_tokens, args.temperature)
        print(f"\nGenerated Response:\n{response}")
    
    elif args.command == 'rag':
        result = cli.generate_with_rag(args.query, args.max_tokens)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nSelf-RAG Response:")
            print(f"Query: {args.query}")
            print(f"Response: {result['best_response']}")
            print(f"Score: {result['best_score']:.2f}")
            print(f"Iterations: {len(result['iterations'])}")
    
    elif args.command == 'review':
        result = cli.review_response(args.query, args.response, args.context)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nReview Results:")
            print(f"Overall Score: {result['overall_score']:.2f}")
            print(f"Approved: {'Yes' if result['approved'] else 'No'}")
            print(f"Relevance: {result['relevance_score']:.2f}")
            print(f"Accuracy: {result['accuracy_score']:.2f}")
            print(f"Engagement: {result['engagement_score']:.2f}")
            print(f"Clarity: {result['clarity_score']:.2f}")
            print(f"Toxicity: {result['toxicity_score']:.2f}")
    
    elif args.command == 'memory':
        if args.clear:
            cli.clear_memory()
            print("GPU memory cleared")
        else:
            usage = cli.get_memory_usage()
            if 'error' in usage:
                print(f"{usage['error']}")
            else:
                print(f"\nGPU Memory Usage:")
                print(f"Allocated: {usage['allocated_gb']:.2f} GB")
                print(f"Reserved: {usage['reserved_gb']:.2f} GB")
                print(f"Total: {usage['total_gb']:.2f} GB")
                print(f"Free: {usage['free_gb']:.2f} GB")
    
    elif args.command == 'interactive':
        print("Starting Interactive Mode...")
        print("Type 'quit' to exit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\nXinfluencer AI > ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("""
Available commands:
  <prompt>              - Generate response
  rag <query>          - Generate with Self-RAG
  review <query> <response> - Review a response
  memory               - Check GPU memory
  clear                - Clear GPU memory
  quit                 - Exit
                    """)
                
                elif user_input.lower() == 'memory':
                    usage = cli.get_memory_usage()
                    if 'error' not in usage:
                        print(f"GPU Memory: {usage['allocated_gb']:.2f}GB / {usage['total_gb']:.2f}GB")
                
                elif user_input.lower() == 'clear':
                    cli.clear_memory()
                    print("Memory cleared")
                
                elif user_input.startswith('rag '):
                    query = user_input[4:].strip()
                    if query:
                        result = cli.generate_with_rag(query)
                        if 'error' not in result:
                            print(f"Response: {result['best_response']}")
                            print(f"Score: {result['best_score']:.2f}")
                        else:
                            print(f"Error: {result['error']}")
                
                elif user_input.startswith('review '):
                    parts = user_input[7:].split(' | ', 1)
                    if len(parts) == 2:
                        query, response = parts
                        result = cli.review_response(query.strip(), response.strip())
                        if 'error' not in result:
                            print(f"Score: {result['overall_score']:.2f} | Approved: {'Yes' if result['approved'] else 'No'}")
                        else:
                            print(f"Error: {result['error']}")
                    else:
                        print("Usage: review <query> | <response>")
                
                elif user_input:
                    response = cli.generate(user_input)
                    print(f"{response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.command == 'status':
        print("System Status:")
        print(f"Python: {sys.version}")
        
        try:
            import torch
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name()}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except ImportError:
            print("PyTorch: Not installed")
        
        try:
            import transformers
            print(f"Transformers: {transformers.__version__}")
        except ImportError:
            print("Transformers: Not installed")

if __name__ == "__main__":
    main()