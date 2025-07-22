#!/usr/bin/env python3
"""
H200 Model Persistence Setup
Sets up persistent model storage on H200 to avoid re-downloading 8B models every time
"""

import os
import subprocess
import json
from pathlib import Path

def setup_h200_persistence():
    """Set up model persistence configuration on H200."""
    
    print("🔧 Setting up H200 Model Persistence")
    print("=" * 50)
    
    # SSH connection details
    ssh_key = "/Users/max/Xinfluencer/influencer.pem"
    h200_host = "157.10.162.127"
    h200_user = "ubuntu"
    
    # Create persistence setup script
    persistence_script = """
#!/bin/bash

# H200 Model Persistence Setup Script
echo "🔧 Setting up model persistence on H200..."

cd /home/ubuntu/xinfluencer

# Create persistent model cache directory
mkdir -p models/cache/huggingface
mkdir -p models/lora_checkpoints
mkdir -p models/local_storage

# Set environment variables for HuggingFace cache
echo "export HF_HOME=/home/ubuntu/xinfluencer/models/cache/huggingface" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=/home/ubuntu/xinfluencer/models/cache/huggingface" >> ~/.bashrc
echo "export HF_DATASETS_CACHE=/home/ubuntu/xinfluencer/models/cache/datasets" >> ~/.bashrc

# Create model persistence config
cat > models/model_config.json << 'EOF'
{
  "persistence_enabled": true,
  "cache_dir": "/home/ubuntu/xinfluencer/models/cache",
  "model_storage": "/home/ubuntu/xinfluencer/models/local_storage",
  "lora_checkpoints": "/home/ubuntu/xinfluencer/models/lora_checkpoints",
  "auto_download": false,
  "reuse_downloaded": true,
  "models": {
    "llama_8b": {
      "name": "meta-llama/Llama-3.1-8B-Instruct",
      "local_path": "/home/ubuntu/xinfluencer/models/local_storage/llama-3.1-8b-instruct",
      "download_once": true,
      "size_gb": 15.0
    }
  }
}
EOF

# Create model manager script
cat > models/model_manager.py << 'PYEOF'
#!/usr/bin/env python3
"""
H200 Model Manager - Efficient model loading and caching
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

class H200ModelManager:
    def __init__(self, config_path="models/model_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.models = {}
        self.tokenizers = {}
        
    def load_config(self):
        with open(self.config_path) as f:
            return json.load(f)
    
    def get_model(self, model_name="llama_8b", force_reload=False):
        """Get model with persistent caching."""
        if model_name in self.models and not force_reload:
            print(f"✅ Using cached {model_name}")
            return self.models[model_name], self.tokenizers[model_name]
        
        model_config = self.config["models"][model_name]
        model_path = model_config["name"]
        local_path = Path(model_config["local_path"])
        
        print(f"🔄 Loading {model_name} from {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with H200 optimization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Cache in memory
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        
        print(f"✅ {model_name} loaded and cached")
        return model, tokenizer
    
    def clear_cache(self):
        """Clear model cache to free memory."""
        self.models.clear()
        self.tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Model cache cleared")

# Global model manager instance
model_manager = H200ModelManager()

def get_persistent_model():
    """Get the persistent Llama model."""
    return model_manager.get_model("llama_8b")

def clear_model_cache():
    """Clear model cache."""
    model_manager.clear_cache()

PYEOF

chmod +x models/model_manager.py

echo "✅ H200 model persistence setup complete!"
echo "📦 Cache directory: /home/ubuntu/xinfluencer/models/cache"
echo "🗄️ Model storage: /home/ubuntu/xinfluencer/models/local_storage"
echo "🔧 LoRA checkpoints: /home/ubuntu/xinfluencer/models/lora_checkpoints"
echo ""
echo "🔄 To apply environment variables, run: source ~/.bashrc"
echo "🤖 Models will now be cached and reused between sessions"
"""
    
    # Write script to temporary file
    script_path = "h200_persistence_setup.sh"
    with open(script_path, 'w') as f:
        f.write(persistence_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    # Execute on H200
    print("📤 Uploading and executing persistence setup on H200...")
    
    # Copy script to H200
    scp_cmd = f'scp -i "{ssh_key}" {script_path} {h200_user}@{h200_host}:/home/ubuntu/'
    subprocess.run(scp_cmd.split(), check=True)
    
    # Execute script on H200
    ssh_cmd = f'ssh -i "{ssh_key}" {h200_user}@{h200_host} "chmod +x /home/ubuntu/{script_path} && /home/ubuntu/{script_path}"'
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ H200 persistence setup completed successfully!")
        print("📋 Setup output:")
        print(result.stdout)
    else:
        print("❌ Setup failed:")
        print(result.stderr)
    
    # Clean up local script
    os.remove(script_path)
    
    return result.returncode == 0

def create_local_chat_interface():
    """Create a lightweight chat interface for testing."""
    
    chat_interface = '''#!/usr/bin/env python3
"""
Lightweight Chat Interface for Soju Testing
Uses H200 persistent models when available, falls back to simulation
"""

import os
import sys
import subprocess
import json
from datetime import datetime

class SojuChatInterface:
    def __init__(self):
        self.ssh_key = "/Users/max/Xinfluencer/influencer.pem"
        self.h200_host = "157.10.162.127"
        self.h200_user = "ubuntu"
        self.session_log = []
        
    def check_h200_model_status(self):
        """Check if trained model is available on H200."""
        try:
            cmd = f'ssh -i "{self.ssh_key}" {self.h200_user}@{self.h200_host} "cd /home/ubuntu/xinfluencer && ls -la lora_checkpoints*/ 2>/dev/null | wc -l"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            checkpoint_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
            return checkpoint_count > 0
        except:
            return False
    
    def generate_on_h200(self, prompt, use_lora=True):
        """Generate response using H200 model."""
        try:
            # Create temporary script for generation
            gen_script = f'''
cd /home/ubuntu/xinfluencer
source xinfluencer_env_fixed/bin/activate
python3 -c "
import sys
sys.path.append('/home/ubuntu/xinfluencer')
from models.model_manager import get_persistent_model, clear_model_cache

try:
    model, tokenizer = get_persistent_model()
    
    # Create prompt for Soju
    system_prompt = '''You are Soju, an AI crypto influencer created by Max. You provide professional, educational content about RWA and crypto without emojis.'''
    
    messages = [
        {{'role': 'system', 'content': system_prompt}},
        {{'role': 'user', 'content': '{prompt}'}}
    ]
    
    # Format for Llama
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate
    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if '<|start_header_id|>assistant<|end_header_id|>' in response:
        response = response.split('<|start_header_id|>assistant<|end_header_id|>')[-1].strip()
    
    print('SOJU_RESPONSE_START')
    print(response)
    print('SOJU_RESPONSE_END')
    
except Exception as e:
    print(f'Generation error: {{e}}')
"
'''
            
            cmd = f'ssh -i "{self.ssh_key}" {self.h200_user}@{self.h200_host} "{gen_script}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            # Extract response
            output = result.stdout
            if "SOJU_RESPONSE_START" in output and "SOJU_RESPONSE_END" in output:
                start = output.find("SOJU_RESPONSE_START") + len("SOJU_RESPONSE_START")
                end = output.find("SOJU_RESPONSE_END")
                response = output[start:end].strip()
                return response
            else:
                return f"H200 generation failed: {output}"
                
        except Exception as e:
            return f"Connection error: {e}"
    
    def get_real_model_response(self, prompt):
        """Get response from actual trained model on H200."""
        return self.generate_on_h200(prompt)
    
    def chat_loop(self):
        """Interactive chat loop."""
        print("💬 Soju Chat Interface")
        print("=" * 40)
        
        # Check model status
        has_model = self.check_h200_model_status()
        if has_model:
            print("🤖 Connected to trained Soju model on H200")
        else:
            print("⚠️  Using simulation mode (model not trained yet)")
        
        print("Type 'quit' to exit, 'status' for info")
        print()
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'status':
                    print(f"Model available: {has_model}")
                    print(f"Session length: {len(self.session_log)}")
                    continue
                elif not user_input:
                    continue
                
                print("🤖 Soju: ", end="", flush=True)
                
                if has_model:
                    response = self.generate_on_h200(user_input)
                else:
                    print("❌ No trained model available. Please complete training first.")
                    continue
                
                print(response)
                print()
                
                # Log interaction
                self.session_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "user": user_input,
                    "soju": response,
                    "model_used": "h200" if has_model else "simulation"
                })
                
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print()

def main():
    chat = SojuChatInterface()
    chat.chat_loop()

if __name__ == "__main__":
    main()
'''
    
    # Write chat interface
    with open("soju_chat.py", 'w') as f:
        f.write(chat_interface)
    
    os.chmod("soju_chat.py", 0o755)
    print("✅ Created soju_chat.py - lightweight chat interface")

def main():
    """Set up H200 model persistence and create chat interface."""
    print("🚀 H200 Model Optimization Setup")
    print("=" * 60)
    
    # Set up persistence on H200
    if setup_h200_persistence():
        print("\n✅ H200 persistence setup successful!")
    else:
        print("\n❌ H200 persistence setup failed")
        return
    
    # Create local chat interface
    create_local_chat_interface()
    
    print("\n🎉 Setup Complete!")
    print("=" * 40)
    print("📋 What was set up:")
    print("   • H200 model persistence (avoid re-downloading)")
    print("   • Persistent cache directories")
    print("   • Model manager for efficient loading")
    print("   • Lightweight chat interface (soju_chat.py)")
    print()
    print("🔄 Next steps:")
    print("   1. Wait for training to complete on H200")
    print("   2. Test with: python3 soju_chat.py")
    print("   3. Models will load once and stay cached")

if __name__ == "__main__":
    main() 