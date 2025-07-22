"""Text generation using language models optimized for H200 GPU."""

from typing import List, Dict, Optional
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from huggingface_hub import login
import logging
import gc

logger = logging.getLogger(__name__)

class TextGenerator:
    """Text generation using causal language models optimized for H200 GPU."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", use_quantization: bool = True):
        """Initialize the generation model with H200 optimization."""
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Authenticate with Hugging Face if token is available
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        if hf_token:
            try:
                login(token=hf_token)
                logger.info("Successfully authenticated with Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to authenticate with Hugging Face: {e}")
        else:
            logger.warning("No HF_TOKEN found in environment - may not be able to access gated models")
        
        logger.info(f"Loading model: {model_name} on {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for H200
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization for memory efficiency")
        else:
            quantization_config = None
            logger.info("Using full precision model")
        
        # Load model with optimization
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load Llama model '{self.model_name}': {e}")
            raise RuntimeError(f"Llama model loading failed: {e}")
        
        logger.info(f"Successfully loaded: {self.model_name}")
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate a response to a prompt with H200 optimization."""
        try:
            # Tokenize input with truncation
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            # Generation config optimized for H200
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Generate with memory optimization
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
            
            # Clear cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.debug(f"Generated response: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_with_context(self, query: str, context: str, max_new_tokens: int = 100) -> str:
        """Generate response using retrieved context with Llama 3.1 formatting."""
        # Create prompt with context using Llama 3.1 format
        if "llama" in self.model_name.lower() and "instruct" in self.model_name.lower():
            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nContext: {context}\n\nQuestion: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        return self.generate_response(prompt, max_new_tokens)
    
    def batch_generate(self, prompts: List[str], max_new_tokens: int = 100) -> List[str]:
        """Generate responses for multiple prompts with memory management."""
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating response {i+1}/{len(prompts)}")
            response = self.generate_response(prompt, max_new_tokens)
            responses.append(response)
            
            # Clear cache between batches
            if torch.cuda.is_available() and i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        logger.info(f"Generated {len(responses)} responses")
        return responses
    
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
        return {"error": -1.0}
    
    def clear_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cleared") 