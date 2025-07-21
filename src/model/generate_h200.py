"""Text generation using Llama 3.1 8B optimized for H200 GPU."""

from typing import List, Dict, Optional
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from huggingface_hub import login
import logging
import gc

logger = logging.getLogger(__name__)

class H200TextGenerator:
    """Text generation using Llama 3.1 8B optimized for H200 GPU."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", use_quantization: bool = False):
        """Initialize the generation model with H200 optimization."""
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Authenticate with Hugging Face
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if requested (optional with 150GB memory)
        quantization_config = None
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")
        else:
            logger.info("Loading model in full precision (recommended for H200)")
        
        # Load model with H200 optimization
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                token=hf_token,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
            
            if not use_quantization and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.1f} GB, Reserved: {memory_reserved:.1f} GB")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
        
        # Set generation config optimized for crypto content
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info("H200TextGenerator initialized successfully")
    
    def generate_response(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """Generate text response using Llama 3.1 8B."""
        try:
            # Format prompt for Llama 3.1 Instruct
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate with optimized settings
            generation_config = GenerationConfig(
                max_new_tokens=min(max_length, 512),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                do_sample=kwargs.get('do_sample', True),
                repetition_penalty=kwargs.get('repetition_penalty', 1.1),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    use_cache=True
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
                response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                response = full_response[len(formatted_prompt):].strip()
            
            logger.info(f"Generated response length: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_crypto_content(self, topic: str, style: str = "informative") -> str:
        """Generate crypto-specific content with style control."""
        style_prompts = {
            "informative": f"Explain {topic} in a clear and informative way for crypto enthusiasts.",
            "bullish": f"Write a positive analysis of {topic} highlighting opportunities.",
            "analytical": f"Provide a balanced technical analysis of {topic}.",
            "educational": f"Create educational content about {topic} for beginners.",
            "news": f"Write a news-style update about {topic}."
        }
        
        prompt = style_prompts.get(style, style_prompts["informative"])
        return self.generate_response(prompt)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts efficiently."""
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, **kwargs)
            responses.append(response)
        return responses
    
    def get_model_info(self) -> Dict:
        """Get model information and current GPU usage."""
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "quantization": self.use_quantization,
            "vocab_size": len(self.tokenizer),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
        
        return info
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleared GPU cache")

# Backwards compatibility
TextGenerator = H200TextGenerator