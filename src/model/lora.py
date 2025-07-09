"""LoRA (Low-Rank Adaptation) fine-tuning implementation."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)

class LoRAFineTuner:
    """LoRA fine-tuning for language models."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize LoRA fine-tuner."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,  # Alpha parameter
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"]  # Safe for DialoGPT
        )
        
        self.peft_model = None
        logger.info(f"LoRA fine-tuner initialized for {model_name}")
    
    def prepare_model_for_training(self):
        """Prepare model with LoRA adapters."""
        self.peft_model = get_peft_model(self.base_model, self.lora_config)
        self.peft_model.to(self.device)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        logger.info("Model prepared with LoRA adapters")
        
        return self.peft_model
    
    def prepare_training_data(self, feedback_data: List[Dict]) -> List[Dict]:
        """Prepare training data from feedback."""
        training_examples = []
        
        for item in feedback_data:
            if item.get("approved", False):  # Only use approved responses
                # Create training example
                prompt = f"Query: {item['query']}\nResponse: {item['response']}"
                
                # Tokenize
                tokenized = self.tokenizer(
                    prompt,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                training_examples.append({
                    "input_ids": tokenized["input_ids"].squeeze(),
                    "attention_mask": tokenized["attention_mask"].squeeze(),
                    "labels": tokenized["input_ids"].squeeze()  # For causal LM
                })
        
        logger.info(f"Prepared {len(training_examples)} training examples")
        return training_examples
    
    def fine_tune(self, training_data: List[Dict], output_dir: str = "./lora_checkpoints") -> str:
        """Fine-tune model with LoRA."""
        if self.peft_model is None:
            self.prepare_model_for_training()
        
        # Create dataset
        dataset = LoRADataset(training_data)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Fine-tune
        logger.info("Starting LoRA fine-tuning...")
        trainer.train()
        
        # Save the adapter
        adapter_path = f"{output_dir}/final_adapter"
        self.peft_model.save_pretrained(adapter_path)
        logger.info(f"LoRA adapter saved to {adapter_path}")
        
        return adapter_path
    
    def load_adapter(self, adapter_path: str):
        """Load a trained LoRA adapter."""
        from peft import PeftModel
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Load with adapter
        self.peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        self.peft_model.to(self.device)
        
        logger.info(f"LoRA adapter loaded from {adapter_path}")
    
    def generate_with_lora(self, prompt: str, max_length: int = 150) -> str:
        """Generate text using LoRA-adapted model."""
        if self.peft_model is None:
            logger.warning("No LoRA model loaded, using base model")
            model = self.base_model
        else:
            model = self.peft_model
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove original prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
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
    """Dataset for LoRA training."""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx] 