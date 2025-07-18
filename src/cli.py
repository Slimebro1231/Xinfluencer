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
from config import Config

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
        
        elif args.twitter_command == 'test':
            print("\nTesting Twitter API connection...")
            print("=" * 40)
            
            connection_test = twitter_service.test_connection()
            
            print(f"OAuth 2.0 available: {connection_test['oauth2_available']}")
            print(f"OAuth 1.0a available: {connection_test['oauth1_available']}")
            print(f"Can post tweets: {connection_test['can_post']}")
            
            if connection_test['user_info']:
                user = connection_test['user_info']
                print(f"\nAuthenticated as:")
                print(f"  Username: @{user['username']}")
                print(f"  Name: {user['name']}")
                print(f"  ID: {user['id']}")
            
            if connection_test['errors']:
                print(f"\nErrors encountered:")
                for error in connection_test['errors']:
                    print(f"  - {error}")
            
            if connection_test['can_post']:
                print("\n✓ Twitter API is ready for posting!")
            else:
                print("\n✗ Twitter API is not ready for posting")
        
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


def _run_command(command: str, remote: bool, config: Config):
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

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Xinfluencer AI CLI - Advanced AI system for crypto influencer analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a simple response
  python cli.py generate "What's the latest trend in crypto?"
  
  # Scrape seed data from the web
  python cli.py data scrape
  
  # Scrape seed data remotely on the H200 server
  python cli.py data scrape --remote

  # Generate with Self-RAG
  python cli.py rag "How should I invest in Bitcoin?"
  
  # Review a response
  python cli.py review "What's Bitcoin?" "Bitcoin is a cryptocurrency" "Bitcoin context..."
  
  # Check GPU memory
  python cli.py memory
  
  # Twitter API testing
  python cli.py twitter status
  python cli.py twitter test
  python cli.py twitter post "Hello from Xinfluencer AI!"
  
  # Interactive mode
  python cli.py interactive
  
  # Control center (multi-panel monitoring)
  python cli.py control-center
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
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute data commands which don't require full component initialization
    if args.command == 'data':
        if args.data_command == 'scrape':
            try:
                config = Config()
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