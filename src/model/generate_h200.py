"""H200-optimized text generation using Mistral model."""

from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import logging
import gc
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class H200TextGenerator:
    """H200-optimized text generation using Llama-3.1-8B-Instruct."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", use_quantization: bool = True):
        """Initialize the H200-optimized generation model."""
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        logger.info(f"🚀 Initializing H200 TextGenerator with {model_name}")
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"🔧 Quantization: {use_quantization}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🎯 GPU: {gpu_name}")
            logger.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        self._load_model()
    
    def _load_model(self):
        """Load the model with H200 optimizations and error handling."""
        try:
            # Load tokenizer first
            logger.info("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir="models"
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ Tokenizer loaded successfully")
            
            # Configure quantization for H200
            quantization_config = None
            if self.use_quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("🔧 Using 4-bit quantization for memory efficiency")
            
            # Load model with H200 optimization
            logger.info("🧠 Loading Llama-3.1 model (this may take a few minutes)...")
            load_start = time.time()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir="models"
            )
            
            load_time = time.time() - load_start
            logger.info(f"✅ Model loaded successfully in {load_time:.1f} seconds")
            
            # Move to device if needed
            if not torch.cuda.is_available():
                self.model.to(self.device)
            
            # Clear memory after loading
            self._clear_memory()
            
            # Log memory usage
            if torch.cuda.is_available():
                memory = self.get_memory_usage()
                logger.info(f"💾 Memory after loading: {memory['allocated_gb']:.1f}/{memory['total_gb']:.1f} GB")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_name}: {e}")
            self._fallback_model()
    
    def _fallback_model(self):
        """Load a fallback model if Llama-3.1 fails."""
        try:
            logger.warning("🔄 Attempting fallback to smaller model...")
            fallback_model = "microsoft/DialoGPT-medium"
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            self.model.to(self.device)
            
            self.model_name = fallback_model
            logger.info(f"✅ Fallback model loaded: {fallback_model}")
            
        except Exception as e:
            logger.error(f"❌ Fallback model also failed: {e}")
            self.model = None
            self.tokenizer = None
    
    def _clear_memory(self):
        """Clear GPU memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate a response with H200 optimization."""
        if self.model is None or self.tokenizer is None:
            return "❌ Model not loaded properly"
        
        try:
            # Format prompt for Llama-3.1 if needed
            if "llama" in self.model_name.lower():
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif "mistral" in self.model_name.lower():
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt
            
            # Tokenize with length limits for H200
            inputs = self.tokenizer.encode(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Conservative limit for H200
            ).to(self.device)
            
            # Generation config optimized for H200
            generation_config = GenerationConfig(
                max_new_tokens=min(max_new_tokens, 200),  # Conservative limit
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Generate with memory management
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Extract just the new generation
            if "llama" in self.model_name.lower():
                # For Llama-3.1, extract content after assistant header
                if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                    response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                else:
                    response = full_response[len(formatted_prompt):].strip()
            elif "mistral" in self.model_name.lower():
                # For Mistral, extract content after [/INST]
                if "[/INST]" in full_response:
                    response = full_response.split("[/INST]")[-1].strip()
                else:
                    response = full_response[len(formatted_prompt):].strip()
            else:
                # For other models, remove the original prompt
                response = full_response[len(formatted_prompt):].strip()
            
            # Clean up memory
            del inputs, outputs
            self._clear_memory()
            
            logger.debug(f"Generated: {response[:100]}...")
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error("🔥 CUDA out of memory! Clearing cache and retrying...")
            self._clear_memory()
            return "❌ GPU memory error - try with shorter prompt or smaller max_tokens"
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"❌ Error: {str(e)}"
    
    def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Alias for generate_response for compatibility."""
        return self.generate_response(prompt, max_new_tokens=max_tokens, temperature=temperature)
    
    def generate_with_context(self, query: str, context: str, max_new_tokens: int = 100) -> str:
        """Generate response using retrieved context."""
        if "mistral" in self.model_name.lower():
            prompt = f"Context: {context}\n\nQuestion: {query}"
        else:
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        return self.generate_response(prompt, max_new_tokens)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = total - reserved
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": free,
                "usage_percent": (allocated / total) * 100
            }
        return {"error": "CUDA not available"}
    
    def clear_memory(self):
        """Public method to clear GPU memory."""
        self._clear_memory()
        logger.info("🧹 GPU memory cleared")
    
    def health_check(self) -> Dict[str, any]:
        """Perform a health check of the model."""
        health = {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "cuda_available": torch.cuda.is_available(),
            "model_name": self.model_name
        }
        
        if torch.cuda.is_available():
            health.update(self.get_memory_usage())
        
        # Quick generation test
        try:
            test_response = self.generate_response("Test", max_new_tokens=10)
            health["generation_test"] = "passed" if test_response and not test_response.startswith("❌") else "failed"
        except Exception as e:
            health["generation_test"] = f"failed: {e}"
        
        return health