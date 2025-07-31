"""Simple LoRA implementation for testing basic functionality."""

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
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

logger = logging.getLogger(__name__)

class SimpleLoRA:
    """Simple LoRA implementation for testing."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """Initialize simple LoRA trainer."""
        self.model_name = model_name
        
        # Get HuggingFace token
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with RoPE scaling fix
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
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
                
                # Try to load cached fixed config first
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                cached_config_path = os.path.join(cache_dir, "fixed_rope_config.json")
                
                if os.path.exists(cached_config_path):
                    logger.info("Loading cached fixed RoPE config...")
                    try:
                        with open(cached_config_path, 'r') as f:
                            config_dict = json.load(f)
                        config = LlamaConfig.from_dict(config_dict)
                        self.model = AutoModelForCausalLM.from_pretrained(
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
                    self.model = AutoModelForCausalLM.from_pretrained(
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
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        token=self.hf_token,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                raise e
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Create PEFT model
        self.peft_model = get_peft_model(self.model, self.lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info(f"Simple LoRA initialized for {model_name}")
    
    def prepare_simple_data(self, feedback_data: List[Dict]) -> List[Dict]:
        """Prepare simple training data."""
        training_data = []
        
        for item in feedback_data:
            if item.get("approved", False):
                tweet_content = item['response']
                
                # Simple format: just the tweet
                training_data.append({
                    "text": tweet_content
                })
        
        logger.info(f"Prepared {len(training_data)} simple training examples")
        return training_data
    
    def train_simple(self, training_data: List[Dict], output_dir: str = "./simple_lora_checkpoints") -> str:
        """Train using simple approach."""
        logger.info(f"Starting simple LoRA training with {len(training_data)} examples")
        
        # Create dataset
        dataset = Dataset.from_list(training_data)
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512
            )
            # Add labels for language modeling (same as input_ids)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            weight_decay=0.01,
            max_grad_norm=1.0,
            fp16=False,
            evaluation_strategy="no"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting simple LoRA training...")
        trainer.train()
        
        # Save
        adapter_path = f"{output_dir}/final_adapter"
        self.peft_model.save_pretrained(adapter_path)
        logger.info(f"Simple LoRA training completed. Adapter saved to {adapter_path}")
        
        return adapter_path
    
    def load_adapter(self, adapter_path: str):
        """Load a trained adapter."""
        logger.info(f"Loading adapter from: {adapter_path}")
        from peft import PeftModel
        
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        logger.info("Adapter loaded successfully")
    
    def generate_simple(self, prompt: str) -> str:
        """Generate text using simple approach."""
        if self.peft_model is None:
            logger.warning("No adapter loaded, using base model")
            model = self.model
        else:
            model = self.peft_model
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            model.eval()
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])),
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response 