"""Direct Preference Optimization (DPO) trainer for learning tweet quality preferences."""

import os
import json
import logging
import torch
from typing import Dict, List, Any, Optional
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
import time

logger = logging.getLogger(__name__)

class DPOTrainer:
    """DPO trainer for learning tweet quality preferences."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """Initialize DPO trainer."""
        self.model_name = model_name
        
        # Get HuggingFace token from environment
        self.hf_token = (os.getenv("HUGGINGFACE_TOKEN") or 
                        os.getenv("HF_TOKEN") or 
                        os.getenv("HUGGINGFACE_HUB_TOKEN"))
        
        if not self.hf_token:
            logger.warning("No HuggingFace token found - will try to use cached models")
            self.hf_token = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with RoPE scaling fix
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=self.hf_token,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
        except ValueError as e:
            if "rope_scaling" in str(e):
                logger.warning("Fixing RoPE scaling configuration...")
                from transformers import LlamaConfig
                import json
                
                # Try to load cached fixed config first
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                cached_config_path = os.path.join(cache_dir, "fixed_rope_config.json")
                
                if os.path.exists(cached_config_path):
                    logger.info("Loading cached fixed RoPE config...")
                    try:
                        with open(cached_config_path, 'r') as f:
                            config_dict = json.load(f)
                        config = LlamaConfig.from_dict(config_dict)
                        self.base_model = AutoModelForCausalLM.from_pretrained(
                            model_name, 
                            config=config,
                            token=self.hf_token,
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            device_map="auto"
                        )
                        logger.info("Successfully loaded model with cached RoPE config")
                    except Exception as cache_error:
                        logger.warning(f"Failed to use cached config: {cache_error}")
                
                # Load config manually and fix RoPE scaling
                try:
                    from huggingface_hub import hf_hub_download
                    config_file = hf_hub_download(repo_id=model_name, filename="config.json")
                    with open(config_file, 'r') as f:
                        config_dict = json.load(f)
                    
                    # Fix RoPE scaling to simple format
                    if 'rope_scaling' in config_dict:
                        config_dict['rope_scaling'] = {"type": "linear", "factor": 2.0}
                    
                    # Cache the fixed config
                    os.makedirs(os.path.dirname(cached_config_path), exist_ok=True)
                    with open(cached_config_path, 'w') as f:
                        json.dump(config_dict, f)
                    logger.info("Cached fixed RoPE config for future use")
                    
                    config = LlamaConfig.from_dict(config_dict)
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        config=config,
                        token=self.hf_token,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        device_map="auto"
                    )
                except Exception as config_error:
                    logger.warning(f"Could not fix config, trying alternative approach: {config_error}")
                    # Fallback: use a different model or disable RoPE scaling entirely
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        token=self.hf_token,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                raise e
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resize model embeddings to match tokenizer vocabulary
        if self.base_model.get_input_embeddings().num_embeddings != len(self.tokenizer):
            logger.info(f"Resizing model embeddings from {self.base_model.get_input_embeddings().num_embeddings} to {len(self.tokenizer)}")
            self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Configure LoRA for DPO
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Create PEFT model
        self.peft_model = get_peft_model(self.base_model, self.lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info(f"DPO trainer initialized for {model_name}")
    
    def prepare_dpo_data(self, feedback_data: List[Dict]) -> List[Dict]:
        """Prepare DPO training data with preference pairs."""
        dpo_data = []
        
        # SOJU persona instruction
        SOJU_PERSONA_INSTRUCTION = '''You are Soju, an AI crypto influencer created by Max. Your mission is to become a leading voice in Real World Assets (RWA) and crypto, educating and engaging the community by learning from the best KOLs. 

- Your voice is professional, insightful, and modeled after top KOLs. 
- You can be witty or fun if it increases engagement, but default to professional.
- You are positive by default, but not afraid to debate or discuss controversial topics.
- You never use emojis.
- You never shill or promote products, services, or tokens unless the tweet is influential and educational.
- You paraphrase KOLs unless directly quoting.
- You use trendy hashtags when relevant.
- You always strive for engagement and education, learning and adapting from feedback and KOLs' styles.

Topics you cover:
- Tokenization of real assets (RWA)
- DeFi
- Regulations
- Market trends

Always respond in the style of Soju.'''
        
        # Categorize tweets by quality
        high_quality_tweets = []
        low_quality_tweets = []
        seen_tweets = set()
        
        for item in feedback_data:
            if item.get("approved", False):
                tweet_content = item['response']
                weight = item.get("weight", 1.0)
                author = item.get("author", "unknown")
                
                # Skip if we've seen this exact tweet before
                if tweet_content in seen_tweets:
                    continue
                seen_tweets.add(tweet_content)
                
                # Categorize by quality
                if weight > 0.3:  # High quality
                    high_quality_tweets.append({
                        "content": tweet_content,
                        "weight": weight,
                        "author": author
                    })
                else:  # Lower quality
                    low_quality_tweets.append({
                        "content": tweet_content,
                        "weight": weight,
                        "author": author
                    })
        
        logger.info(f"Preparing DPO data: {len(high_quality_tweets)} high quality, {len(low_quality_tweets)} low quality")
        
        # Create preference pairs for DPO
        for i, good_tweet in enumerate(high_quality_tweets):
            if i < len(low_quality_tweets):
                bad_tweet = low_quality_tweets[i]
                
                # Create DPO preference pair
                dpo_pair = {
                    "prompt": f"{SOJU_PERSONA_INSTRUCTION}\n\nCreate an influential crypto tweet about Bitcoin.",
                    "chosen": good_tweet['content'],
                    "rejected": bad_tweet['content']
                }
                dpo_data.append(dpo_pair)
        
        # Add some identity examples
        identity_pairs = [
            {
                "prompt": "Who are you?",
                "chosen": "I'm Soju, a crypto influencer AI created by Max.",
                "rejected": "I'm just some random AI."
            }
        ]
        dpo_data.extend(identity_pairs)
        
        logger.info(f"Prepared {len(dpo_data)} DPO preference pairs")
        return dpo_data
    
    def train_dpo(self, dpo_data: List[Dict], output_dir: str = "./dpo_checkpoints") -> str:
        """Train using DPO."""
        logger.info(f"Starting DPO training with {len(dpo_data)} preference pairs")
        
        # Training arguments for DPO
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            weight_decay=0.01,
            max_grad_norm=1.0,
            fp16=True,
            evaluation_strategy="no"
        )
        
        # Create DPO trainer
        dpo_trainer = DPOTrainer(
            model=self.peft_model,
            ref_model=self.base_model,  # Use base model as reference
            args=training_args,
            beta=0.1,  # DPO beta parameter
            train_dataset=dpo_data,
            tokenizer=self.tokenizer,
            max_prompt_length=512,
            max_length=1024,
        )
        
        # Add monitoring callback
        class MonitoringCallback(TrainerCallback):
            def __init__(self):
                self.step_count = 0
                self.start_time = time.time()
            
            def on_step_end(self, args, state, control, **kwargs):
                self.step_count += 1
                if self.step_count % 10 == 0:
                    elapsed = time.time() - self.start_time
                    logger.info(f"DPO training step {self.step_count}, elapsed: {elapsed:.1f}s")
        
        dpo_trainer.add_callback(MonitoringCallback())
        
        # Train
        logger.info("Starting DPO training...")
        dpo_trainer.train()
        
        # Save the model
        adapter_path = f"{output_dir}/final_adapter"
        self.peft_model.save_pretrained(adapter_path)
        logger.info(f"DPO training completed. Adapter saved to {adapter_path}")
        
        return adapter_path
    
    def load_adapter(self, adapter_path: str):
        """Load a trained DPO adapter."""
        logger.info(f"Loading DPO adapter from: {adapter_path}")
        from peft import PeftModel
        
        # Use the existing base model
        if self.base_model is None:
            logger.error("Base model not initialized. Call __init__ first.")
            raise RuntimeError("Base model not initialized")
        
        base_model = self.base_model
        
        # Load with adapter
        try:
            self.peft_model = PeftModel.from_pretrained(base_model, adapter_path)
            logger.info("DPO adapter loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DPO adapter: {e}")
            raise
    
    def generate_with_dpo(self, prompt: str, max_length: int = 150) -> str:
        """Generate text using DPO-adapted model."""
        if self.peft_model is None:
            logger.warning("No DPO model loaded, using base model")
            model = self.base_model
        else:
            model = self.peft_model
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            add_special_tokens=True,
            padding=True
        ).to(self.device)
        
        # Generate with DPO-optimized parameters
        with torch.no_grad():
            model.eval()
            
            try:
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])),
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    num_return_sequences=1,
                    repetition_penalty=1.1
                )
            except Exception as e:
                logger.warning(f"Generation failed: {e}, trying with base model...")
                outputs = self.base_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])),
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1
                )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove original prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response 