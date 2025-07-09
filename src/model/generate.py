"""Text generation using language models optimized for H200 GPU."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Optional
import gc

class TextGenerator:
    """A text generator that uses a Hugging Face model with H200 optimization."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1", device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer with H200 optimization."""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("Model loaded successfully.")

    def generate(self, prompt: str, max_length: int = 150, temperature: float = 0.7) -> str:
        """Generate text from a prompt."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                num_return_sequences=1,
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