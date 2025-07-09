#!/usr/bin/env python3
"""
Xinfluencer AI CLI
A comprehensive command-line interface for the AI system
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model.generate import TextGenerator
from vector.search import VectorSearcher
from vector.db import VectorDB
from vector.embed import TextEmbedder
from model.selfrag import SelfRAGGenerator
from review.ai import AIReviewer
from utils.logger import setup_logger

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
  
  # Generate with Self-RAG
  python cli.py rag "How should I invest in Bitcoin?"
  
  # Review a response
  python cli.py review "What's Bitcoin?" "Bitcoin is a cryptocurrency" "Bitcoin context..."
  
  # Check GPU memory
  python cli.py memory
  
  # Interactive mode
  python cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
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
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = XinfluencerCLI()
    
    # Initialize components if needed
    if args.command in ['generate', 'rag', 'review', 'interactive']:
        use_gpu = not getattr(args, 'no_gpu', False)
        if not cli.initialize_components(use_gpu):
            print("❌ Failed to initialize AI components")
            return
    
    # Execute commands
    if args.command == 'generate':
        response = cli.generate(args.prompt, args.max_tokens, args.temperature)
        print(f"\n🤖 Generated Response:\n{response}")
    
    elif args.command == 'rag':
        result = cli.generate_with_rag(args.query, args.max_tokens)
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n🧠 Self-RAG Response:")
            print(f"Query: {args.query}")
            print(f"Response: {result['best_response']}")
            print(f"Score: {result['best_score']:.2f}")
            print(f"Iterations: {len(result['iterations'])}")
    
    elif args.command == 'review':
        result = cli.review_response(args.query, args.response, args.context)
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"\n📋 Review Results:")
            print(f"Overall Score: {result['overall_score']:.2f}")
            print(f"Approved: {'✅' if result['approved'] else '❌'}")
            print(f"Relevance: {result['relevance_score']:.2f}")
            print(f"Accuracy: {result['accuracy_score']:.2f}")
            print(f"Engagement: {result['engagement_score']:.2f}")
            print(f"Clarity: {result['clarity_score']:.2f}")
            print(f"Toxicity: {result['toxicity_score']:.2f}")
    
    elif args.command == 'memory':
        if args.clear:
            cli.clear_memory()
            print("🧹 GPU memory cleared")
        else:
            usage = cli.get_memory_usage()
            if 'error' in usage:
                print(f"❌ {usage['error']}")
            else:
                print(f"\n💾 GPU Memory Usage:")
                print(f"Allocated: {usage['allocated_gb']:.2f} GB")
                print(f"Reserved: {usage['reserved_gb']:.2f} GB")
                print(f"Total: {usage['total_gb']:.2f} GB")
                print(f"Free: {usage['free_gb']:.2f} GB")
    
    elif args.command == 'interactive':
        print("🚀 Starting Interactive Mode...")
        print("Type 'quit' to exit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\n🤖 Xinfluencer AI > ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
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
                        print(f"💾 GPU Memory: {usage['allocated_gb']:.2f}GB / {usage['total_gb']:.2f}GB")
                
                elif user_input.lower() == 'clear':
                    cli.clear_memory()
                    print("🧹 Memory cleared")
                
                elif user_input.startswith('rag '):
                    query = user_input[4:].strip()
                    if query:
                        result = cli.generate_with_rag(query)
                        if 'error' not in result:
                            print(f"🧠 Response: {result['best_response']}")
                            print(f"📊 Score: {result['best_score']:.2f}")
                        else:
                            print(f"❌ Error: {result['error']}")
                
                elif user_input.startswith('review '):
                    parts = user_input[7:].split(' | ', 1)
                    if len(parts) == 2:
                        query, response = parts
                        result = cli.review_response(query.strip(), response.strip())
                        if 'error' not in result:
                            print(f"📋 Score: {result['overall_score']:.2f} | Approved: {'✅' if result['approved'] else '❌'}")
                        else:
                            print(f"❌ Error: {result['error']}")
                    else:
                        print("❌ Usage: review <query> | <response>")
                
                elif user_input:
                    response = cli.generate(user_input)
                    print(f"🤖 {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif args.command == 'status':
        print("🔍 System Status:")
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