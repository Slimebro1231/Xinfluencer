#!/usr/bin/env python3
"""
Manual Training Script
Bypasses HuggingFace Trainer for better control and debugging.
"""

import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import signal
import sys

logger = logging.getLogger(__name__)

class ManualTrainingDataset(Dataset):
    """Custom dataset for manual training."""
    
    def __init__(self, training_data: List[Dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logger.info(f"Creating dataset with {len(training_data)} training examples")
        
        for i, item in enumerate(training_data):
            # Format the training example
            if 'instruction' in item and 'response' in item:
                # Instruction-following format
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['response']}<|end_of_text|>"
            elif 'query' in item and 'response' in item:
                # Query-response format (from identity pipeline)
                text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{item['query']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{item['response']}<|end_of_text|>"
            elif 'text' in item:
                # Text is already formatted for instruction-following
                text = item['text']
                # Ensure it ends with end_of_text token
                if not text.endswith('<|end_of_text|>'):
                    text += '<|end_of_text|>'
            else:
                logger.warning(f"Skipping item {i} - no valid text field found. Keys: {list(item.keys())}")
                continue
            
            try:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,
                    return_tensors="pt"
                )
                
                # Create labels (same as input_ids for causal LM)
                inputs['labels'] = inputs['input_ids'].clone()
                
                self.data.append(inputs)
                
                if i < 3:  # Log first few examples
                    logger.info(f"Example {i}: Text length {len(text)}, Tokenized length {len(inputs['input_ids'][0])}")
                    
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class ManualTrainer:
    """Manual trainer that bypasses HuggingFace Trainer."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.should_stop = False
        self.model = None
        self.base_model = None
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Fix tokenizer parallelism warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Get HuggingFace token
        self.hf_token = (os.getenv("HUGGINGFACE_TOKEN") or 
                        os.getenv("HF_TOKEN") or 
                        os.getenv("HUGGINGFACE_HUB_TOKEN"))
        
        if not self.hf_token:
            logger.warning("No HuggingFace token found - will try to use cached models")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with fixed RoPE configuration
        self._load_model_with_fixed_rope()
        
        # Initialize LoRA
        self._setup_lora()
        
        logger.info(f"Manual trainer initialized on device: {self.device}")
    
    def _load_model_with_fixed_rope(self):
        """Load model with fixed RoPE configuration."""
        try:
            # Load config and fix RoPE scaling
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(repo_id=self.model_name, filename="config.json")
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Fix RoPE scaling configuration
            if 'rope_scaling' in config_dict:
                config_dict['rope_scaling'] = {"type": "linear", "factor": 2.0}
                logger.info("Fixed RoPE scaling configuration")
            
            config = LlamaConfig.from_dict(config_dict)
            
            # Load model with fixed config
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                token=self.hf_token,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.base_model = self.base_model.to(self.device)
            logger.info("Successfully loaded model with fixed RoPE config")
            
        except Exception as e:
            logger.error(f"Failed to load model with fixed config: {e}")
            # Fallback
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.base_model = self.base_model.to(self.device)
            logger.info("Successfully loaded model with fallback method")
    
    def _setup_lora(self):
        """Setup LoRA configuration."""
        # Use working LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
        
        # Create PEFT model
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model = self.model.to(self.device)
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Ensure LoRA parameters require gradients
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                logger.info(f"Set {name} to require gradients")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        logger.info("LoRA setup completed")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.info(f"Received signal {signum}, stopping training gracefully...")
        self.should_stop = True
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.model is not None:
                del self.model
            if self.base_model is not None:
                del self.base_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Cleaned up model resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def prepare_training_data(self, training_data: List[Dict]) -> ManualTrainingDataset:
        """Prepare training data."""
        logger.info(f"Preparing dataset with {len(training_data)} training examples")
        logger.info(f"First example keys: {list(training_data[0].keys()) if training_data else 'No data'}")
        
        dataset = ManualTrainingDataset(training_data, self.tokenizer)
        logger.info(f"Prepared {len(dataset)} training examples")
        return dataset
    
    def train(self, 
              training_data: List[Dict], 
              output_dir: str = "lora_checkpoints/manual_training",
              epochs: int = 3,
              batch_size: int = 2,
              learning_rate: float = 1e-4,
              save_steps: int = 100) -> str:
        """Manual training loop."""
        
        logger.info(f"Starting manual training with {len(training_data)} examples")
        logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
        
        # Prepare dataset
        dataset = self.prepare_training_data(training_data)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0
        
        try:
            for epoch in range(epochs):
                if self.should_stop:
                    logger.info("Training stopped by user request")
                    break
                    
                epoch_loss = 0
                epoch_steps = 0
                
                logger.info(f"Starting epoch {epoch + 1}/{epochs}")
                
                for batch_idx, batch in enumerate(dataloader):
                    if self.should_stop:
                        logger.info("Training stopped by user request")
                        break
                        
                    try:
                        # Move batch to device
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        # Forward pass
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        # Update metrics
                        epoch_loss += loss.item()
                        total_loss += loss.item()
                        epoch_steps += 1
                        global_step += 1
                        
                        # Log progress
                        if batch_idx % 10 == 0:
                            avg_loss = epoch_loss / epoch_steps
                            logger.info(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
                        
                        # Save checkpoint
                        if global_step % save_steps == 0:
                            self._save_checkpoint(output_dir, global_step)
                            
                    except Exception as e:
                        logger.error(f"Error in training step {global_step}: {e}")
                        continue
                
                if self.should_stop:
                    break
                    
                # End of epoch
                avg_epoch_loss = epoch_loss / epoch_steps
                logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
                
                # Save epoch checkpoint
                self._save_checkpoint(output_dir, f"epoch_{epoch + 1}")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.should_stop = True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup
            self.cleanup()
        
        # Save final model
        final_path = self._save_checkpoint(output_dir, "final")
        
        logger.info(f"Training completed! Final model saved to: {final_path}")
        logger.info(f"Total training steps: {global_step}")
        logger.info(f"Final average loss: {total_loss / global_step:.4f}")
        
        return final_path
    
    def _collate_fn(self, batch):
        """Custom collate function for variable length sequences."""
        # Find max length in batch
        max_length = max(len(item['input_ids'][0]) for item in batch)
        
        # Pad sequences
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            seq_len = len(item['input_ids'][0])
            
            # Pad input_ids
            padded_input = item['input_ids'][0].tolist() + [self.tokenizer.pad_token_id] * (max_length - seq_len)
            input_ids.append(padded_input)
            
            # Pad attention_mask
            padded_attention = item['attention_mask'][0].tolist() + [0] * (max_length - seq_len)
            attention_mask.append(padded_attention)
            
            # Pad labels (use -100 for padding tokens)
            padded_labels = item['labels'][0].tolist() + [-100] * (max_length - seq_len)
            labels.append(padded_labels)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }
    
    def _save_checkpoint(self, output_dir: str, step: str) -> str:
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(output_dir, f"checkpoint_{step}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training info
        info = {
            'step': step,
            'timestamp': time.time(),
            'model_name': self.model_name,
            'lora_config': self.lora_config.to_dict()
        }
        
        with open(os.path.join(checkpoint_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Checkpoint saved to: {checkpoint_dir}")
        return checkpoint_dir
    
    def load_adapter(self, adapter_path: str):
        """Load a trained adapter."""
        self.model = self.model.from_pretrained(adapter_path)
        self.model = self.model.to(self.device)
        logger.info(f"Adapter loaded from: {adapter_path}")
    
    def generate(self, prompt: str, max_length: int = 150) -> str:
        """Generate text with the trained model."""
        self.model.eval()
        
        # Format prompt for Llama 3.1
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant response
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            response = generated_text
        
        return response 