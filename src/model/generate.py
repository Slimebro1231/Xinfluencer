"""Text generation using language models."""

from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import logging

logger = logging.getLogger(__name__)

class TextGenerator:
    """Text generation using causal language models."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize the generation model."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loaded generation model: {model_name} on {self.device}")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate a response to a prompt."""
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # Generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                generation_config=generation_config,
                return_dict_in_generate=True
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Remove original prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        logger.debug(f"Generated response: {response[:50]}...")
        return response
    
    def generate_with_context(self, query: str, context: str, max_new_tokens: int = 50) -> str:
        """Generate response using retrieved context."""
        # Create prompt with context
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        return self.generate_response(prompt, max_new_tokens)
    
    def batch_generate(self, prompts: List[str], max_new_tokens: int = 50) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, max_new_tokens)
            responses.append(response)
        
        logger.info(f"Generated {len(responses)} responses")
        return responses 