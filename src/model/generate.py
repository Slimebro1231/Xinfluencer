"""Text generation using language models optimized for H200 GPU."""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Optional
import gc
from config import config

class TextGenerator:
    """Handles text generation using a Hugging Face model."""

    def __init__(self, model_name: str = None):
        """Initialize the generator with a model name."""
        self.model_name = model_name or config.model.generation_model
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer from Hugging Face."""
        if not self.model or not self.tokenizer:
            print(f"Loading model: {self.model_name}")

            hf_token = config.model.huggingface_token
            if not hf_token:
                raise ValueError("Hugging Face token not found. Please set HUGGINGFACE_TOKEN in your .env file.")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                device_map="auto"
            )
            print("Model loaded successfully.")

    def generate(self, prompt: str, max_length: int = 150, temperature: float = 0.7, max_new_tokens: int = None) -> str:
        """Generate text from a prompt."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set max_new_tokens if provided, otherwise use max_length for backward compatibility
        if max_new_tokens is not None:
            generation_kwargs = {"max_new_tokens": max_new_tokens}
        else:
            generation_kwargs = {"max_length": max_length}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1,
                **generation_kwargs
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": total - reserved
            }
        return {"error": "CUDA not available"}

    def clear_memory(self):
        """Clear GPU memory."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 