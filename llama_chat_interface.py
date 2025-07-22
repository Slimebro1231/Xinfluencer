#!/usr/bin/env python3
"""
Llama-3.1-8B Chat Interface with LoRA Support
Proper model for testing crypto identity training results
"""

import os
import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from pathlib import Path

class LlamaChatInterface:
    def __init__(self, model_path="meta-llama/Llama-3.1-8B-Instruct", lora_path=None, use_4bit=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.lora_path = lora_path
        
        print(f"🦙 Loading Llama on {self.device}")
        print(f"📦 Base model: {model_path}")
        if lora_path:
            print(f"🎯 LoRA adapter: {lora_path}")
        
        # Configure for H200 GPU memory efficiency
        if use_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        print("🔄 Loading Llama model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if specified
        if lora_path and os.path.exists(lora_path):
            try:
                from peft import PeftModel
                print(f"🔄 Loading LoRA adapter from {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                print("✅ LoRA adapter loaded successfully")
                self.has_lora = True
            except ImportError:
                print("❌ PEFT not available, skipping LoRA adapter")
                self.has_lora = False
            except Exception as e:
                print(f"❌ Error loading LoRA adapter: {e}")
                self.has_lora = False
        else:
            self.has_lora = False
            if lora_path:
                print(f"❌ LoRA path not found: {lora_path}")
        
        print("✅ Llama model loaded successfully\n")
    
    def create_crypto_prompt(self, user_message):
        """Create a crypto-expert focused prompt for Llama"""
        system_prompt = """You are a leading crypto and RWA (Real World Assets) expert with deep knowledge of:
- Cryptocurrency markets and technology (Bitcoin, Ethereum, DeFi)
- Real World Asset tokenization and protocols
- Blockchain technology and smart contracts
- Market trends and investment insights
- Technical analysis and fundamental analysis

Provide accurate, insightful, and actionable information. Be concise but comprehensive."""
        
        # Use Llama-3.1 chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Format for Llama-3.1-Instruct
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate_response(self, user_message, max_new_tokens=300, temperature=0.7, top_p=0.9):
        """Generate a response using Llama"""
        prompt = self.create_crypto_prompt(user_message)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            # Fallback: extract everything after the prompt
            response = full_response[len(prompt):].strip()
        
        return response
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("🦙 Llama-3.1-8B Crypto Expert Chat")
        print("=" * 60)
        model_type = "TRAINED (with LoRA)" if self.has_lora else "BASE MODEL"
        print(f"Model: {model_type}")
        print("Ask about crypto, RWA, blockchain, or any related topics.")
        print("Commands: 'quit'/'exit' to stop, 'help' for commands, 'model' for info\n")
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("Commands:")
                    print("  quit/exit - Exit the chat")
                    print("  help - Show this help")
                    print("  model - Show model information")
                    print("  test - Run crypto knowledge test")
                    continue
                elif user_input.lower() == 'model':
                    print(f"Model: {self.model_path}")
                    print(f"Device: {self.device}")
                    print(f"LoRA: {'Yes' if self.has_lora else 'No'}")
                    if self.has_lora:
                        print(f"LoRA path: {self.lora_path}")
                    continue
                elif user_input.lower() == 'test':
                    self.run_crypto_test()
                    continue
                elif not user_input:
                    continue
                
                print("🦙 Llama: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
    
    def run_crypto_test(self):
        """Run a series of crypto knowledge tests"""
        test_questions = [
            "What is RWA tokenization?",
            "How does Bitcoin's Lightning Network work?",
            "What are the benefits of DeFi protocols?",
            "Explain smart contract security best practices"
        ]
        
        print("\n🧪 Crypto Knowledge Test")
        print("-" * 30)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. {question}")
            print("🦙 Response:", end=" ", flush=True)
            response = self.generate_response(question, max_new_tokens=150)
            print(response)
        
        print("\n" + "-" * 30)
        print("Test complete. Compare responses between base and trained models.")

def check_for_lora_checkpoints():
    """Check for existing LoRA checkpoints"""
    possible_paths = [
        "lora_checkpoints/identity",
        "lora_checkpoints/crypto", 
        "data/training_ready/lora_checkpoint",
        "../lora_checkpoints/identity"
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if it has required LoRA files
            required_files = ["adapter_config.json", "adapter_model.bin"]
            if all(os.path.exists(os.path.join(path, f)) for f in required_files):
                return path
    
    return None

def check_training_data():
    """Check available training data"""
    training_paths = [
        "data/training_ready/crypto_training_dataset.json",
        "data/training_ready/real_posts.db"
    ]
    
    print("📊 Training Data Status:")
    for path in training_paths:
        if os.path.exists(path):
            if path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    print(f"   ✅ {path}: {len(data)} training examples")
                except:
                    print(f"   ❌ {path}: exists but unreadable")
            else:
                print(f"   ✅ {path}: exists")
        else:
            print(f"   ❌ {path}: not found")

def main():
    parser = argparse.ArgumentParser(description="Llama-3.1-8B Crypto Expert Chat")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", 
                       help="Llama model to use")
    parser.add_argument("--lora", help="Path to LoRA adapter directory")
    parser.add_argument("--no-4bit", action="store_true", 
                       help="Disable 4-bit quantization")
    parser.add_argument("--single", help="Ask a single question and exit")
    
    args = parser.parse_args()
    
    # Check training data status
    check_training_data()
    
    # Auto-detect LoRA if not specified
    if not args.lora:
        auto_lora = check_for_lora_checkpoints()
        if auto_lora:
            print(f"\n🔍 Found LoRA checkpoint: {auto_lora}")
            use_lora = input("Use this LoRA adapter? (y/n): ").lower().startswith('y')
            if use_lora:
                args.lora = auto_lora
    
    # Initialize chat interface
    try:
        chat = LlamaChatInterface(
            model_path=args.model,
            lora_path=args.lora,
            use_4bit=not args.no_4bit
        )
        
        if args.single:
            # Single question mode
            print(f"\n👤 Question: {args.single}")
            response = chat.generate_response(args.single)
            print(f"🦙 Response: {response}")
        else:
            # Interactive chat mode
            chat.chat_loop()
            
    except Exception as e:
        print(f"❌ Failed to initialize Llama chat: {e}")
        print("Make sure you have proper access to the Llama model.")

if __name__ == "__main__":
    main() 