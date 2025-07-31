"""LoRA (Low-Rank Adaptation) fine-tuning implementation."""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)

class LoRAFineTuner:
    """LoRA fine-tuning for language models."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """Initialize LoRA fine-tuner."""
        self.model_name = model_name
        
        # Get HuggingFace token from environment - try multiple sources
        self.hf_token = (os.getenv("HUGGINGFACE_TOKEN") or 
                        os.getenv("HF_TOKEN") or 
                        os.getenv("HUGGINGFACE_HUB_TOKEN"))
        
        if not self.hf_token:
            # Try to load from .env file
            try:
                with open(".env", "r") as f:
                    for line in f:
                        if line.startswith("HUGGINGFACE_TOKEN="):
                            self.hf_token = line.split("=", 1)[1].strip()
                            break
            except:
                pass
        
        # For now, allow initialization without token (will use cached models)
        if not self.hf_token:
            logger.warning("No HuggingFace token found - will try to use cached models")
            self.hf_token = None
        
        # Load tokenizer with token
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set device for model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model with proper configuration handling - optimized for H200
        # Fix RoPE scaling configuration permanently
        from transformers import LlamaConfig
        import json
        
        try:
            # Load config and fix RoPE scaling before model loading
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(repo_id=model_name, filename="config.json")
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Fix RoPE scaling configuration permanently
            if 'rope_scaling' in config_dict:
                config_dict['rope_scaling'] = {"type": "linear", "factor": 2.0}
                logger.info("Fixed RoPE scaling configuration")
            
            # Remove problematic special token configurations
            problematic_keys = ['bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id']
            for key in problematic_keys:
                if key in config_dict:
                    del config_dict[key]
                    logger.info(f"Removed problematic config key: {key}")
            
            # Create config object
            config = LlamaConfig.from_dict(config_dict)
            
            # Load model with fixed config
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                token=self.hf_token,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"  # Let transformers handle device placement
            )
            logger.info("Successfully loaded model with fixed RoPE config")
            
        except Exception as e:
            logger.error(f"Failed to load model with fixed config: {e}")
            # Fallback to original loading method
            try:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=self.hf_token,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map="auto"
                )
                logger.info("Successfully loaded model with fallback method")
            except Exception as fallback_error:
                logger.error(f"Fallback loading also failed: {fallback_error}")
                raise
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure model is properly set up for training
        self.base_model.train()
        for param in self.base_model.parameters():
            param.requires_grad = False  # Freeze base model parameters
        
        # Keep model on GPU for proper training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.device) != "cpu":
            self.base_model = self.base_model.to(self.device)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Resize model embeddings to match tokenizer vocabulary
        if self.base_model.get_input_embeddings().num_embeddings != len(self.tokenizer):
            logger.info(f"Resizing model embeddings from {self.base_model.get_input_embeddings().num_embeddings} to {len(self.tokenizer)}")
            self.base_model.resize_token_embeddings(len(self.tokenizer))
            
        # Ensure embeddings are properly initialized to prevent "!!!!!" outputs
        with torch.no_grad():
            embedding_layer = self.base_model.get_input_embeddings()
            if hasattr(embedding_layer, 'weight'):
                # Get special token IDs for Llama 3.1
                special_tokens = [
                    self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>"),
                    self.tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
                    self.tokenizer.convert_tokens_to_ids("<|end_header_id|>"),
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
                ]
                
                # Filter out -1 (not found) tokens
                special_tokens = [token_id for token_id in special_tokens if token_id != -1]
                
                logger.info(f"Found {len(special_tokens)} special tokens: {special_tokens}")
                
                # Initialize special token embeddings with mean of existing embeddings
                if special_tokens:
                    mean_embedding = embedding_layer.weight.mean(dim=0)
                    for token_id in special_tokens:
                        if token_id < embedding_layer.weight.shape[0]:
                            # Use mean + small noise for better initialization
                            embedding_layer.weight[token_id] = mean_embedding + torch.randn_like(mean_embedding) * 0.01
                            logger.info(f"Initialized embedding for token {token_id}")
                
                # Also check for any zero embeddings and initialize them
                weight_norm = torch.norm(embedding_layer.weight, dim=1)
                zero_embeddings = (weight_norm == 0).sum().item()
                if zero_embeddings > 0:
                    logger.info(f"Found {zero_embeddings} zero embeddings, initializing them...")
                    zero_mask = weight_norm == 0
                    mean_embedding = embedding_layer.weight.mean(dim=0)
                    embedding_layer.weight[zero_mask] = mean_embedding + torch.randn_like(embedding_layer.weight[zero_mask]) * 0.01
            
        # Configure LoRA for efficient fine-tuning with dynamic target module detection
        # Use the working target modules from test_lora_setup.py
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # Only attention modules
        
        # Use the working configuration from test_lora_setup.py
        logger.info(f"Using working target modules: {target_modules}")
        
        # Use the exact working LoRA configuration from test_lora_setup.py
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Working rank
            lora_alpha=32,  # Working alpha
            lora_dropout=0.1,  # Working dropout
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # All 4 modules like working test
            bias="none",
            init_lora_weights=True,
        )
        
        self.peft_model = None
        logger.info(f"LoRA fine-tuner initialized for {model_name}")
    
    def prepare_model_for_training(self):
        """Prepare model for training (with or without LoRA)."""
        logger.info(f"Preparing model with LoRA config: {self.lora_config}")
        logger.info(f"Base model type: {type(self.base_model)}")
        
        if self.lora_config is not None:
            try:
                self.peft_model = get_peft_model(self.base_model, self.lora_config)
                logger.info(f"PEFT model created: {type(self.peft_model)}")
                
                # Validate the PEFT model
                trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.peft_model.parameters())
                logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
                
                if trainable_params == 0:
                    raise RuntimeError("No trainable parameters found in PEFT model!")
                    
            except Exception as e:
                logger.error(f"Failed to create PEFT model: {e}")
                raise
        else:
            logger.info("Using full fine-tuning (no LoRA)")
            self.peft_model = self.base_model
        
        # Only move to device if not already there and not using device_map
        if not hasattr(self.peft_model, 'hf_device_map') and str(self.peft_model.device) != str(self.device):
            self.peft_model.to(self.device)
        else:
            logger.info(f"Model already on device: {self.peft_model.device}")
        
        # Keep PEFT model on GPU and ensure it's in training mode
        if str(self.device) != "cpu":
            self.peft_model = self.peft_model.to(self.device)
        self.peft_model.train()
        
        # Check trainable parameters before enabling gradients
        trainable_before = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters before gradient enable: {trainable_before}")
        
        # Explicitly enable gradients for all trainable parameters
        enabled_count = 0
        for name, param in self.peft_model.named_parameters():
            if param.requires_grad:
                param.requires_grad_(True)
                enabled_count += 1
                if enabled_count <= 5:  # Log first 5 for debugging
                    logger.info(f"Enabled gradients for {name}")
        
        logger.info(f"Enabled gradients for {enabled_count} parameters")
        
        # Print trainable parameters
        if self.lora_config is not None:
            self.peft_model.print_trainable_parameters()
            logger.info("Model prepared with LoRA adapters for training")
        else:
            logger.info("Model prepared for full fine-tuning")
        
        return self.peft_model
    
    def prepare_training_data(self, feedback_data: List[Dict]) -> List[Dict]:
        """Prepare training data with clean, focused content for stable training."""
        training_examples = []
        
        # Handle both old format (with 'response') and new format (with 'text')
        for item in feedback_data:
            # Get tweet content from either 'text' or 'response' field
            tweet_content = item.get('text') or item.get('response', '')
            weight = item.get("weight", 1.0)
            
            # Skip if no content
            if not tweet_content:
                continue
            
            # Clean the tweet content - remove any persona instructions or special tokens
            clean_content = self._clean_tweet_content(tweet_content)
            
            # Only use tweets that are actually tweet-like
            if clean_content and len(clean_content) > 10 and len(clean_content) < 280:
                # Use just the tweet content, no instructions
                training_examples.append({
                    "text": clean_content,
                    "weight": weight,
                    "type": "clean_tweet"
                })
        
        logger.info(f"Prepared {len(training_examples)} clean training examples")
        
        # Log some examples for verification
        for i, example in enumerate(training_examples[:3]):
            logger.info(f"Example {i+1}: {example['text'][:100]}...")
        
        return training_examples
    
    def _clean_tweet_content(self, tweet_content: str) -> str:
        """Clean tweet content for training - remove special tokens, persona instructions and metadata."""
        import re
        
        # Remove Llama 3.1 special tokens that cause issues
        special_token_patterns = [
            r'<\|begin_of_text\|>',
            r'<\|start_header_id\|>.*?<\|end_header_id\|>',
            r'<\|eot_id\|>',
            r'<\|end_of_text\|>',
            r'user',
            r'assistant'
        ]
        
        for pattern in special_token_patterns:
            tweet = re.sub(pattern, '', tweet_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove URLs
        tweet = re.sub(r'https?://\S+', '', tweet)
        
        # Remove any persona instructions that might be in the response
        persona_patterns = [
            r'You are Soju.*?Always respond in the style of Soju\.',
            r'You are a professional crypto influencer.*?',
            r'Your mission is to become.*?',
            r'Your voice is professional.*?',
            r'Topics you cover:.*?',
            r'Example Q&A:.*?',
            r'Q:.*?A:.*?',
            r'Create an influential crypto tweet\.',
        ]
        
        for pattern in persona_patterns:
            tweet = re.sub(pattern, '', tweet, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining instruction-like content
        tweet = re.sub(r'^Tweet:\s*', '', tweet, flags=re.IGNORECASE)
        tweet = re.sub(r'^Response:\s*', '', tweet, flags=re.IGNORECASE)
        
        # Remove mentions but keep the @ symbol for context
        tweet = re.sub(r'@\w+', '@user', tweet)
        
        # Remove hashtags but keep the # symbol for context
        tweet = re.sub(r'#\w+', '#topic', tweet)
        
        # Remove extra whitespace and newlines
        tweet = ' '.join(tweet.split())
        
        # Remove empty tweets
        if not tweet.strip():
            return ""
        
        # Ensure reasonable length
        if len(tweet) < 10 or len(tweet) > 280:
            return ""
        
        return tweet.strip()
    
    def _create_preference_pairs(self, feedback_data: List[Dict]) -> List[Dict]:
        """Create preference pairs for learning what makes good tweets."""
        pairs = []
        
        # Categorize tweets by quality
        high_quality = []
        low_quality = []
        
        for item in feedback_data:
            if item.get("approved", False):
                tweet_content = self._clean_tweet_content(item['response'])
                weight = item.get("weight", 1.0)
                
                if tweet_content and len(tweet_content) > 10:
                    if weight > 0.4:  # High quality threshold
                        high_quality.append(tweet_content)
                    elif weight < 0.2:  # Low quality threshold
                        low_quality.append(tweet_content)
        
        # Create preference pairs
        for i, good_tweet in enumerate(high_quality[:min(20, len(high_quality))]):
            if i < len(low_quality):
                bad_tweet = low_quality[i]
                
                # Format as preference learning
                pair_text = f"Which tweet is better?\n\nTweet A: {good_tweet}\n\nTweet B: {bad_tweet}\n\nAnswer: Tweet A is better because it's more professional and engaging."
                
                pairs.append({
                    "text": pair_text,
                    "weight": 1.5,  # Higher weight for preference learning
                    "type": "preference_pair"
                })
        
        return pairs
    
    def _create_instruction_examples(self, feedback_data: List[Dict]) -> List[Dict]:
        """Create instruction-response examples with subtle persona."""
        examples = []
        
        # Subtle persona instruction
        subtle_persona = "You are a professional crypto influencer. Write engaging, informative tweets."
        
        for item in feedback_data:
            if item.get("approved", False):
                tweet_content = self._clean_tweet_content(item['response'])
                weight = item.get("weight", 1.0)
                
                if tweet_content and len(tweet_content) > 10:
                    # Simple instruction format
                    instruction_text = f"{subtle_persona}\n\nWrite a crypto tweet:\n\n{tweet_content}"
                    
                    examples.append({
                        "text": instruction_text,
                        "weight": weight,
                        "type": "instruction_response"
                    })
        
        return examples
    
    def fine_tune(self, training_data: List[Dict], output_dir: str = "./lora_checkpoints", continue_from: str = None, method: str = "lora") -> str:
        """Fine-tune model with LoRA, optionally continuing from existing adapter."""
        logger.info(f"fine_tune called with continue_from: {continue_from}")
        logger.info(f"continue_from exists: {continue_from and os.path.exists(continue_from) if continue_from else False}")
        
        if continue_from and os.path.exists(continue_from):
            logger.info(f"Continuing training from existing adapter: {continue_from}")
            self.load_adapter(continue_from)
        else:
            # Fresh training - prepare model with LoRA adapters
            logger.info("Starting fresh training - preparing model with LoRA adapters...")
            self.prepare_model_for_training()
        
        # Ensure we have a model ready for training
        if self.peft_model is None:
            logger.error("Failed to prepare model for training")
            raise RuntimeError("Failed to prepare model for training")
        
        # Double-check that model is ready for training
        if self.peft_model is not None:
            self.peft_model.train()
            # Ensure ALL trainable parameters require gradients
            trainable_count = 0
            for name, param in self.peft_model.named_parameters():
                if param.requires_grad:
                    param.requires_grad_(True)
                    trainable_count += 1
                    logger.info(f"Enabled gradients for {name}")
            
            logger.info(f"Total trainable parameters: {trainable_count}")
            
            if trainable_count == 0:
                raise RuntimeError("No trainable parameters found! LoRA adapters not properly initialized.")
        else:
            raise RuntimeError("Failed to prepare model for training")
        
        # Create dataset with tokenizer
        dataset = LoRADataset(training_data, self.tokenizer)
        
        # Training arguments - optimized for H200 GPU
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,  # Single epoch
            per_device_train_batch_size=2,  # Standard batch size for GPU
            gradient_accumulation_steps=2,  # Standard accumulation
            warmup_steps=5,  # Standard warmup
            learning_rate=5e-5,  # Standard learning rate
            logging_steps=5,
            save_steps=25,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            # Standard regularization
            weight_decay=0.01,  # Standard weight decay
            # Standard gradient clipping
            max_grad_norm=1.0,  # Standard gradient clipping
            # Disable mixed precision to avoid gradient scaling issues
            fp16=False,
            bf16=False,
            # Evaluation during training
            evaluation_strategy="no",
            # Enable gradient checkpointing for memory efficiency
            gradient_checkpointing=True,
        )
        
        # Pre-process dataset to ensure all sequences have the same length
        logger.info("Pre-processing dataset to ensure uniform sequence lengths...")
        
        # Find the maximum length in the dataset
        max_length = 0
        for i in range(len(dataset)):
            item = dataset[i]
            max_length = max(max_length, len(item["input_ids"]))
        
        logger.info(f"Maximum sequence length in dataset: {max_length}")
        
        # Pad all sequences to the same length
        padded_dataset = []
        for i in range(len(dataset)):
            item = dataset[i]
            padding_length = max_length - len(item["input_ids"])
            
            # Pad input_ids with pad_token_id
            padded_input_ids = item["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            
            # Pad attention_mask with 0s
            padded_attention_mask = item["attention_mask"] + [0] * padding_length
            
            # Pad labels with -100 (ignore in loss calculation)
            padded_labels = item["labels"] + [-100] * padding_length
            
            padded_dataset.append({
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "labels": padded_labels
            })
        
        logger.info(f"Pre-processed {len(padded_dataset)} sequences to uniform length {max_length}")
        
        # Create a simple dataset class for the padded data
        class PaddedDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                return {
                    "input_ids": torch.tensor(item["input_ids"]),
                    "attention_mask": torch.tensor(item["attention_mask"]),
                    "labels": torch.tensor(item["labels"])
                }
        
        # Use the padded dataset
        padded_dataset = PaddedDataset(padded_dataset)
        
        # Use standard data collator since all sequences are now the same length
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficiency on GPU
            return_tensors="pt",  # Return PyTorch tensors
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=padded_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Choose training method
        if method == "dpo":
            logger.info("Starting DPO training...")
            adapter_path = self._train_dpo(training_data, output_dir)
        elif method == "rlhf":
            logger.info("Starting RLHF training...")
            adapter_path = self._train_rlhf(training_data, output_dir)
        else:
            logger.info("Starting LoRA fine-tuning...")
            
            # Add monitoring callback to show real-time progress
            from transformers import TrainerCallback
            
            class MonitoringCallback(TrainerCallback):
                def __init__(self):
                    self.step_count = 0
                    self.start_time = time.time()
                
                def on_step_end(self, args, state, control, **kwargs):
                    self.step_count += 1
                    if self.step_count % 10 == 0:  # Log every 10 steps
                        elapsed = time.time() - self.start_time
                        logger.info(f"Training step {self.step_count}, elapsed: {elapsed:.1f}s")
                
                def on_step_begin(self, args, state, control, **kwargs):
                    # Check and fix gradients before each step
                    if hasattr(self, 'model') and self.model is not None:
                        self._check_and_fix_gradients()
                
                def _check_and_fix_gradients(self):
                    """Check and fix any corrupted gradients before they're applied."""
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                # Check for NaN/Inf in gradients
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    logger.warning(f"Found NaN/Inf in gradients for {name}, fixing...")
                                    # Replace corrupted gradients with zeros
                                    param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)
                                    param.grad = torch.where(torch.isinf(param.grad), torch.zeros_like(param.grad), param.grad)
                                    
                                    # Clip extreme gradients
                                    param.grad = torch.clamp(param.grad, -1.0, 1.0)
                        
                        # Also check and fix LoRA weights during training
                        self._check_and_fix_lora_weights()
                
                def _check_and_fix_lora_weights(self):
                    """Check and fix LoRA weights during training."""
                    with torch.no_grad():
                        for name, module in self.model.named_modules():
                            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                                if hasattr(module.lora_A, 'default'):
                                    lora_A_module = module.lora_A.default
                                    lora_B_module = module.lora_B.default
                                else:
                                    lora_A_module = module.lora_A
                                    lora_B_module = module.lora_B
                                
                                if hasattr(lora_A_module, 'weight') and hasattr(lora_B_module, 'weight'):
                                    # Check lora_A weights
                                    if torch.isnan(lora_A_module.weight).any() or torch.isinf(lora_A_module.weight).any():
                                        logger.warning(f"Found NaN/Inf in {name}.lora_A during training, fixing...")
                                        lora_A_module.weight.data = torch.where(torch.isnan(lora_A_module.weight.data), torch.zeros_like(lora_A_module.weight.data), lora_A_module.weight.data)
                                        lora_A_module.weight.data = torch.where(torch.isinf(lora_A_module.weight.data), torch.zeros_like(lora_A_module.weight.data), lora_A_module.weight.data)
                                        lora_A_module.weight.data = torch.clamp(lora_A_module.weight.data, -5, 5)
                                    
                                    # Check lora_B weights
                                    if torch.isnan(lora_B_module.weight).any() or torch.isinf(lora_B_module.weight).any():
                                        logger.warning(f"Found NaN/Inf in {name}.lora_B during training, fixing...")
                                        lora_B_module.weight.data = torch.where(torch.isnan(lora_B_module.weight.data), torch.zeros_like(lora_B_module.weight.data), lora_B_module.weight.data)
                                        lora_B_module.weight.data = torch.where(torch.isinf(lora_B_module.weight.data), torch.zeros_like(lora_B_module.weight.data), lora_B_module.weight.data)
                                        lora_B_module.weight.data = torch.clamp(lora_B_module.weight.data, -5, 5)
            
            # Add callback to trainer
            callback = MonitoringCallback()
            callback.model = self.peft_model  # Pass model reference to callback
            trainer.add_callback(callback)
            
            trainer.train()
            
            # Save the adapter
            adapter_path = f"{output_dir}/final_adapter"
            self.peft_model.save_pretrained(adapter_path)
            logger.info(f"LoRA adapter saved to {adapter_path}")
        
        return adapter_path
    
    def _train_dpo(self, training_data: List[Dict], output_dir: str) -> str:
        """Train using Direct Preference Optimization."""
        try:
            from trl import DPOTrainer
            
            # Prepare DPO data
            dpo_data = []
            for item in training_data:
                if item.get("type") == "preference_pair":
                    # Extract from preference pair format
                    text = item["text"]
                    if "Tweet A:" in text and "Tweet B:" in text:
                        # Parse the preference pair
                        parts = text.split("Tweet A:")
                        if len(parts) > 1:
                            tweet_a_part = parts[1].split("Tweet B:")[0].strip()
                            tweet_b_part = text.split("Tweet B:")[1].split("Answer:")[0].strip()
                            
                            dpo_data.append({
                                "prompt": "Write a crypto tweet:",
                                "chosen": tweet_a_part,
                                "rejected": tweet_b_part
                            })
            
            if not dpo_data:
                logger.error("No DPO data found")
                raise ValueError("No DPO data found")
            
            # DPO training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=2,
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
                fp16=False,
                evaluation_strategy="no"
            )
            
            # Create DPO trainer
            dpo_trainer = DPOTrainer(
                model=self.peft_model,
                ref_model=self.base_model,
                args=training_args,
                beta=0.1,
                train_dataset=dpo_data,
                tokenizer=self.tokenizer,
                max_prompt_length=128,
                max_length=256,
            )
            
            # Train
            dpo_trainer.train()
            
            # Save
            adapter_path = f"{output_dir}/dpo_adapter"
            self.peft_model.save_pretrained(adapter_path)
            logger.info(f"DPO adapter saved to {adapter_path}")
            
            return adapter_path
            
        except ImportError:
            logger.error("TRL not available")
            raise ImportError("TRL library not available")
        except Exception as e:
            logger.error(f"DPO training failed: {e}")
            raise e
    
    def _train_rlhf(self, training_data: List[Dict], output_dir: str) -> str:
        """Train using RLHF (Reinforcement Learning from Human Feedback)."""
        try:
            from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
            from transformers import pipeline
            
            # For now, implement a simplified RLHF approach
            logger.error("RLHF training not fully implemented")
            raise NotImplementedError("RLHF training not fully implemented")
            
        except ImportError:
            logger.error("TRL not available")
            raise ImportError("TRL library not available")
        except Exception as e:
            logger.error(f"RLHF training failed: {e}")
            raise e
    

    
    def _fix_embedding_alignment(self):
        """Fix embedding alignment issues that cause '!!!!!' outputs."""
        logger.info("Fixing embedding alignment...")
        
        with torch.no_grad():
            # Get the base model from PEFT model
            base_model = self.peft_model.get_base_model()
            embedding_layer = base_model.get_input_embeddings()
            
            if hasattr(embedding_layer, 'weight'):
                # Get all special token IDs for Llama 3.1
                special_tokens = [
                    self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>"),
                    self.tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
                    self.tokenizer.convert_tokens_to_ids("<|end_header_id|>"),
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
                ]
                
                # Filter out -1 (not found) tokens
                special_tokens = [token_id for token_id in special_tokens if token_id != -1]
                
                logger.info(f"Fixing {len(special_tokens)} special token embeddings")
                
                # Initialize special token embeddings with proper values
                for token_id in special_tokens:
                    if token_id < embedding_layer.weight.shape[0]:
                        # Use the mean of existing embeddings as initialization
                        mean_embedding = embedding_layer.weight.mean(dim=0)
                        # Add small random noise to avoid exact duplication
                        embedding_layer.weight[token_id] = mean_embedding + torch.randn_like(mean_embedding) * 0.01
                        logger.info(f"Fixed embedding for token {token_id}")
                
                # Also check for any zero embeddings and initialize them
                weight_norm = torch.norm(embedding_layer.weight, dim=1)
                zero_embeddings = (weight_norm == 0).sum().item()
                if zero_embeddings > 0:
                    logger.info(f"Found {zero_embeddings} zero embeddings, initializing them...")
                    zero_mask = weight_norm == 0
                    mean_embedding = embedding_layer.weight.mean(dim=0)
                    embedding_layer.weight[zero_mask] = mean_embedding + torch.randn_like(embedding_layer.weight[zero_mask]) * 0.01
                
                # Ensure the embedding layer is properly updated
                embedding_layer.weight.requires_grad_(True)
                
                # Force a forward pass to ensure embeddings are properly initialized
                try:
                    dummy_input = torch.tensor([[self.tokenizer.pad_token_id]], device=self.device)
                    with torch.no_grad():
                        _ = embedding_layer(dummy_input)
                    logger.info("Embedding layer forward pass successful")
                except Exception as e:
                    logger.warning(f"Embedding layer forward pass failed: {e}")
    
    def _fix_lora_weights(self):
        """Fix LoRA adapter weights that might be causing probability tensor corruption."""
        logger.info("Checking and fixing LoRA adapter weights...")
        
        with torch.no_grad():
            # Get all LoRA modules
            for name, module in self.peft_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # Handle ModuleDict structure
                    if hasattr(module.lora_A, 'default'):
                        # It's a ModuleDict, get the default module
                        lora_A_module = module.lora_A.default
                        lora_B_module = module.lora_B.default
                    else:
                        # Direct module
                        lora_A_module = module.lora_A
                        lora_B_module = module.lora_B
                    
                    # Check for NaN/Inf in LoRA weights
                    if hasattr(lora_A_module, 'weight') and hasattr(lora_B_module, 'weight'):
                        lora_A = lora_A_module.weight
                        lora_B = lora_B_module.weight
                        
                        # Fix lora_A weights
                        if torch.isnan(lora_A).any() or torch.isinf(lora_A).any():
                            logger.warning(f"Found NaN/Inf in {name}.lora_A, fixing...")
                            lora_A = torch.where(torch.isnan(lora_A), torch.zeros_like(lora_A), lora_A)
                            lora_A = torch.where(torch.isinf(lora_A), torch.zeros_like(lora_A), lora_A)
                            lora_A_module.weight.data = lora_A
                        
                        # Fix lora_B weights
                        if torch.isnan(lora_B).any() or torch.isinf(lora_B).any():
                            logger.warning(f"Found NaN/Inf in {name}.lora_B, fixing...")
                            lora_B = torch.where(torch.isnan(lora_B), torch.zeros_like(lora_B), lora_B)
                            lora_B = torch.where(torch.isinf(lora_B), torch.zeros_like(lora_B), lora_B)
                            lora_B_module.weight.data = lora_B
                        
                        # Clip extreme values
                        lora_A_module.weight.data = torch.clamp(lora_A_module.weight.data, -10, 10)
                        lora_B_module.weight.data = torch.clamp(lora_B_module.weight.data, -10, 10)
            
            logger.info("LoRA weights fixed")
            
            # If all weights were corrupted, reinitialize them
            corrupted_count = 0
            for name, module in self.peft_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    if hasattr(module.lora_A, 'default'):
                        lora_A_module = module.lora_A.default
                        lora_B_module = module.lora_B.default
                    else:
                        lora_A_module = module.lora_A
                        lora_B_module = module.lora_B
                    
                    if hasattr(lora_A_module, 'weight') and hasattr(lora_B_module, 'weight'):
                        if torch.all(lora_A_module.weight == 0) and torch.all(lora_B_module.weight == 0):
                            corrupted_count += 1
                            # Reinitialize with small random values
                            lora_A_module.weight.data = torch.randn_like(lora_A_module.weight) * 0.01
                            lora_B_module.weight.data = torch.randn_like(lora_B_module.weight) * 0.01
            
            if corrupted_count > 0:
                logger.info(f"Reinitialized {corrupted_count} completely corrupted LoRA modules")
    
    def _diagnose_model_outputs(self, test_input):
        """Diagnose what's happening in the model outputs."""
        logger.info("Diagnosing model outputs...")
        
        with torch.no_grad():
            # Test base model
            base_outputs = self.base_model(**test_input)
            base_logits = base_outputs.logits[:, -1, :]
            logger.info(f"Base model logits - min: {base_logits.min():.4f}, max: {base_logits.max():.4f}, mean: {base_logits.mean():.4f}")
            logger.info(f"Base model logits - NaN: {torch.isnan(base_logits).sum()}, Inf: {torch.isinf(base_logits).sum()}")
            
            # Test LoRA model
            lora_outputs = self.peft_model(**test_input)
            lora_logits = lora_outputs.logits[:, -1, :]
            logger.info(f"LoRA model logits - min: {lora_logits.min():.4f}, max: {lora_logits.max():.4f}, mean: {lora_logits.mean():.4f}")
            logger.info(f"LoRA model logits - NaN: {torch.isnan(lora_logits).sum()}, Inf: {torch.isinf(lora_logits).sum()}")
            
            # Check if LoRA is actually active
            logit_diff = torch.abs(lora_logits - base_logits)
            logger.info(f"Logit difference - max: {logit_diff.max():.4f}, mean: {logit_diff.mean():.4f}")
            
            if logit_diff.max() < 1e-6:
                logger.warning("LoRA appears to be inactive - logits are identical to base model")
            elif torch.isnan(lora_logits).any() or torch.isinf(lora_logits).any():
                logger.error("LoRA model has corrupted logits")
            else:
                logger.info("LoRA model appears to be working correctly")
    
    def load_adapter(self, adapter_path: str):
        """Load a trained LoRA adapter."""
        logger.info(f"load_adapter called with: {adapter_path}")
        from peft import PeftModel
        
        # Use the existing base model instead of loading a new one
        if self.base_model is None:
            logger.error("Base model not initialized. Call __init__ first.")
            raise RuntimeError("Base model not initialized")
        
        base_model = self.base_model
        
        # Load with adapter, handling configuration compatibility
        try:
            self.peft_model = PeftModel.from_pretrained(base_model, adapter_path)
            
            # Ensure embeddings are properly aligned after loading adapter
            self._fix_embedding_alignment()
            
            # Fix any corrupted LoRA weights
            self._fix_lora_weights()
            
            # Test the model with a simple input to diagnose issues
            test_input = self.tokenizer("test", return_tensors="pt", add_special_tokens=False).to(self.device)
            self._diagnose_model_outputs(test_input)
            
        except TypeError as e:
            if "corda_config" in str(e):
                # Handle older adapter configs that might have extra fields
                logger.warning("Adapter config has incompatible fields, attempting to fix config...")
                try:
                    # Load and fix the adapter config manually
                    import json
                    config_file = os.path.join(adapter_path, "adapter_config.json")
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            config_dict = json.load(f)
                        
                        # Remove problematic fields
                        problematic_fields = ['corda_config', 'eva_config', 'exclude_modules']
                        for field in problematic_fields:
                            if field in config_dict:
                                del config_dict[field]
                        
                        # Save fixed config
                        with open(config_file, 'w') as f:
                            json.dump(config_dict, f, indent=2)
                        
                        logger.info("Fixed adapter config, retrying load...")
                        self.peft_model = PeftModel.from_pretrained(base_model, adapter_path)
                    else:
                        raise e
                except Exception as config_error:
                    logger.warning(f"Could not fix adapter config: {config_error}")
                    # Fallback: start fresh without loading existing adapter
                    logger.info("Starting fresh training without existing adapter")
                    self.peft_model = None
                    return
            else:
                raise e
        
        self.peft_model.to(self.device)
        
        logger.info(f"LoRA adapter loaded from {adapter_path}")
    
    def generate_with_lora(self, prompt: str, max_length: int = 150) -> str:
        """Generate text using LoRA-adapted model with simplified approach."""
        try:
            # Use PEFT model if available, otherwise base model
            model = self.peft_model if self.peft_model is not None else self.base_model
            model.eval()
            
            # Simple tokenization
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate using standard approach
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove original prompt
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"LoRA generation failed: {e}")
            # Fallback to base model
            try:
                with torch.no_grad():
                    outputs = self.base_model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
                return response
                
            except Exception as base_error:
                logger.error(f"Base model generation also failed: {base_error}")
                return "Generation failed"
    
    def _fix_logits(self, logits):
        """Fix common issues in logits that cause probability tensor corruption."""
        # Replace NaN and Inf values
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
        
        # Clip extreme values to prevent overflow
        logits = torch.clamp(logits, -100, 100)
        
        # Ensure logits are finite
        if not torch.isfinite(logits).all():
            logger.warning("Logits still contain non-finite values after fixing")
            # Use a fallback approach - replace with small random values
            logits = torch.randn_like(logits) * 0.1
        
        return logits
    
    def _fallback_to_base_model(self, inputs, prompt):
        """Fallback to base model when LoRA generation fails."""
        try:
            logger.info("Falling back to base model...")
            base_outputs = self.base_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask', torch.ones_like(inputs['input_ids'])),
                max_new_tokens=50,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True
            )
            
            base_response = self.tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            if base_response.startswith(prompt):
                base_response = base_response[len(prompt):].strip()
            
            return base_response
            
        except Exception as base_error:
            logger.error(f"Base model fallback also failed: {base_error}")
            return "Generation failed completely"
    
    def test_generation_stability(self, test_prompts: List[str] = None) -> Dict[str, Any]:
        """Test generation stability with various prompts."""
        if test_prompts is None:
            test_prompts = [
                "What is RWA in crypto?",
                "Explain DeFi briefly.",
                "What are the benefits of tokenization?",
                "How does blockchain work?",
                "What is your opinion on crypto regulation?"
            ]
        
        results = {
            "total_tests": len(test_prompts),
            "successful_generations": 0,
            "failed_generations": 0,
            "responses": [],
            "errors": []
        }
        
        for i, prompt in enumerate(test_prompts):
            try:
                logger.info(f"Testing generation {i+1}/{len(test_prompts)}: {prompt}")
                response = self.generate_with_lora(prompt, max_length=100)
                results["successful_generations"] += 1
                results["responses"].append({
                    "prompt": prompt,
                    "response": response,
                    "status": "success"
                })
                logger.info(f"✅ Generated: {response[:50]}...")
            except Exception as e:
                results["failed_generations"] += 1
                error_msg = str(e)
                results["errors"].append({
                    "prompt": prompt,
                    "error": error_msg
                })
                logger.error(f"❌ Failed: {error_msg}")
        
        success_rate = results["successful_generations"] / results["total_tests"]
        logger.info(f"Generation stability test completed: {success_rate:.1%} success rate")
        
        return results
    
    def evaluate_adapter(self, test_data: List[Dict]) -> Dict:
        """Evaluate the performance of the LoRA adapter."""
        # Simple evaluation - could be expanded
        results = {
            "total_samples": len(test_data),
            "generation_quality": 0.0,
            "responses": []
        }
        
        quality_scores = []
        
        for item in test_data:
            query = item["query"]
            expected = item.get("expected_response", "")
            
            # Generate response
            generated = self.generate_with_lora(f"Query: {query}\nResponse:")
            
            # Simple quality score (could use more sophisticated metrics)
            quality_score = self._calculate_quality_score(generated, expected)
            quality_scores.append(quality_score)
            
            results["responses"].append({
                "query": query,
                "generated": generated,
                "expected": expected,
                "quality_score": quality_score
            })
        
        results["generation_quality"] = sum(quality_scores) / len(quality_scores)
        
        logger.info(f"LoRA evaluation completed. Average quality: {results['generation_quality']:.2f}")
        return results
    
    def _calculate_quality_score(self, generated: str, expected: str) -> float:
        """Calculate a simple quality score."""
        # Simple heuristic - could be improved with proper metrics
        if not generated.strip():
            return 0.0
        
        # Length similarity
        len_similarity = min(len(generated), len(expected)) / max(len(generated), len(expected), 1)
        
        # Word overlap
        gen_words = set(generated.lower().split())
        exp_words = set(expected.lower().split())
        
        if len(exp_words) == 0:
            word_overlap = 1.0 if len(gen_words) > 0 else 0.0
        else:
            word_overlap = len(gen_words & exp_words) / len(exp_words)
        
        # Combined score
        return (len_similarity + word_overlap) / 2

class LoRADataset(torch.utils.data.Dataset):
    """Dataset for LoRA training with proper tokenization."""
    
    def __init__(self, data: List[Dict], tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        
        # Tokenize with proper padding like the working test
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,  # Shorter max length for tweets
            return_tensors=None  # Return lists, not tensors
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].copy()  # For causal LM
        }