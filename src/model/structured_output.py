"""Structured output training for consistent, parsable responses."""

from typing import Dict, List, Optional, Tuple
import re
import json
import logging
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

logger = logging.getLogger(__name__)

@dataclass
class StructuredOutput:
    """Structured output format for crypto analysis."""
    area: str
    sources: str
    relevance: str
    message: str
    anticipated_engagement: str
    confidence: float
    evidence_quality: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'area': self.area,
            'sources': self.sources,
            'relevance': self.relevance,
            'message': self.message,
            'anticipated_engagement': self.anticipated_engagement,
            'confidence': self.confidence,
            'evidence_quality': self.evidence_quality
        }
    
    def to_text(self) -> str:
        """Convert to structured text format."""
        return f"""[AREA] {self.area}
[SOURCES] {self.sources}
[RELEVANCE] {self.relevance}
[MESSAGE] {self.message}
[ANTICIPATED_ENGAGEMENT] {self.anticipated_engagement}
[CONFIDENCE] {self.confidence:.2f}
[EVIDENCE_QUALITY] {self.evidence_quality:.2f}"""

class StructuredOutputParser:
    """Parse structured outputs from text."""
    
    def __init__(self):
        self.patterns = {
            'area': r'\[AREA\]\s*(.+)',
            'sources': r'\[SOURCES\]\s*(.+)',
            'relevance': r'\[RELEVANCE\]\s*(.+)',
            'message': r'\[MESSAGE\]\s*(.+)',
            'anticipated_engagement': r'\[ANTICIPATED_ENGAGEMENT\]\s*(.+)',
            'confidence': r'\[CONFIDENCE\]\s*([0-9.]+)',
            'evidence_quality': r'\[EVIDENCE_QUALITY\]\s*([0-9.]+)'
        }
    
    def parse(self, text: str) -> Optional[StructuredOutput]:
        """Parse structured output from text."""
        try:
            matches = {}
            for field, pattern in self.patterns.items():
                match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
                if match:
                    matches[field] = match.group(1).strip()
                else:
                    logger.warning(f"Missing field: {field}")
                    return None
            
            return StructuredOutput(
                area=matches['area'],
                sources=matches['sources'],
                relevance=matches['relevance'],
                message=matches['message'],
                anticipated_engagement=matches['anticipated_engagement'],
                confidence=float(matches['confidence']),
                evidence_quality=float(matches['evidence_quality'])
            )
        except Exception as e:
            logger.error(f"Failed to parse structured output: {e}")
            return None
    
    def validate(self, output: StructuredOutput) -> bool:
        """Validate structured output."""
        try:
            # Check required fields
            if not all([output.area, output.sources, output.relevance, output.message]):
                return False
            
            # Check numeric ranges
            if not (0.0 <= output.confidence <= 1.0):
                return False
            if not (0.0 <= output.evidence_quality <= 1.0):
                return False
            
            # Check engagement format
            if not re.match(r'\d+\s+likes?,\s*\d+\s+retweets?', output.anticipated_engagement):
                return False
            
            return True
        except Exception:
            return False

class StructuredOutputTrainer:
    """Train models to generate structured outputs."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """Initialize structured output trainer."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.parser = StructuredOutputParser()
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Structured output prompt template
        self.structured_prompt_template = """Generate a crypto analysis tweet in the following format:

[AREA] <topic area>
[SOURCES] <data sources>
[RELEVANCE] <why this matters>
[MESSAGE] <main content>
[ANTICIPATED_ENGAGEMENT] <expected likes/retweets>
[CONFIDENCE] <0.0-1.0>
[EVIDENCE_QUALITY] <0.0-1.0>

Query: {query}
Context: {context}

Response:"""
        
        logger.info(f"Structured output trainer initialized for {model_name}")
    
    def create_training_example(self, query: str, context: str, 
                              structured_output: StructuredOutput) -> str:
        """Create a training example with structured output."""
        prompt = self.structured_prompt_template.format(
            query=query,
            context=context
        )
        
        response = structured_output.to_text()
        return prompt + "\n" + response
    
    def create_preference_pairs(self, training_data: List[Dict]) -> List[Tuple[str, str]]:
        """Create preference pairs for DPO training."""
        pairs = []
        
        for item in training_data:
            # Structured output (preferred)
            structured_prompt = self.create_training_example(
                item['query'],
                item['context'],
                item['structured_output']
            )
            
            # Unstructured output (rejected)
            unstructured_prompt = f"Query: {item['query']}\nContext: {item['context']}\nResponse: {item['unstructured_output']}"
            
            pairs.append((structured_prompt, unstructured_prompt))
        
        logger.info(f"Created {len(pairs)} preference pairs for DPO training")
        return pairs
    
    def evaluate_structured_output(self, output: StructuredOutput, 
                                 actual_engagement: Dict) -> float:
        """Evaluate structured output quality."""
        try:
            # Parse anticipated engagement
            anticipated_match = re.match(r'(\d+)\s+likes?,\s*(\d+)\s+retweets?', 
                                       output.anticipated_engagement)
            if not anticipated_match:
                return 0.0
            
            anticipated_likes = int(anticipated_match.group(1))
            anticipated_retweets = int(anticipated_match.group(2))
            
            # Get actual engagement
            actual_likes = actual_engagement.get('likes', 0)
            actual_retweets = actual_engagement.get('retweets', 0)
            
            # Calculate prediction accuracy
            likes_accuracy = 1.0 - min(abs(anticipated_likes - actual_likes) / max(actual_likes, 1), 1.0)
            retweets_accuracy = 1.0 - min(abs(anticipated_retweets - actual_retweets) / max(actual_retweets, 1), 1.0)
            
            # Weighted score
            score = (
                0.3 * output.confidence +
                0.3 * output.evidence_quality +
                0.2 * likes_accuracy +
                0.2 * retweets_accuracy
            )
            
            return score
        except Exception as e:
            logger.error(f"Failed to evaluate structured output: {e}")
            return 0.0
    
    def generate_structured_output(self, model, query: str, context: str) -> Optional[StructuredOutput]:
        """Generate structured output using trained model."""
        try:
            prompt = self.structured_prompt_template.format(
                query=query,
                context=context
            )
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response part
            if "Response:" in full_response:
                response_part = full_response.split("Response:")[-1].strip()
            else:
                response_part = full_response[len(prompt):].strip()
            
            # Parse structured output
            return self.parser.parse(response_part)
            
        except Exception as e:
            logger.error(f"Failed to generate structured output: {e}")
            return None

class StructuredOutputDataset:
    """Dataset for structured output training."""
    
    def __init__(self, data: List[Dict]):
        """Initialize dataset."""
        self.data = data
        self.parser = StructuredOutputParser()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Validate structured output
        if not self.parser.validate(item['structured_output']):
            logger.warning(f"Invalid structured output at index {idx}")
            return None
        
        return item
    
    def filter_valid(self) -> 'StructuredOutputDataset':
        """Filter out invalid examples."""
        valid_data = []
        for item in self.data:
            if self.parser.validate(item['structured_output']):
                valid_data.append(item)
        
        logger.info(f"Filtered dataset: {len(valid_data)}/{len(self.data)} valid examples")
        return StructuredOutputDataset(valid_data)

def create_sample_training_data() -> List[Dict]:
    """Create sample training data for structured outputs."""
    return [
        {
            'query': 'What is the impact of Bitcoin halving on price?',
            'context': 'Bitcoin halving reduces mining rewards by 50%, historically leading to price increases due to supply reduction.',
            'structured_output': StructuredOutput(
                area='Bitcoin Analysis',
                sources='Bitcoin whitepaper, historical halving data, Coinbase metrics',
                relevance='Supply reduction typically leads to price appreciation',
                message='Bitcoin halving reduces daily supply by 50%. Historical data shows this often leads to significant price appreciation within 12-18 months.',
                anticipated_engagement='1500 likes, 300 retweets',
                confidence=0.85,
                evidence_quality=0.92
            ),
            'unstructured_output': 'Bitcoin halving is bullish for price because it reduces supply.',
            'actual_engagement': {'likes': 1420, 'retweets': 285}
        },
        {
            'query': 'How does DeFi lending work?',
            'context': 'DeFi lending platforms use smart contracts to enable peer-to-peer lending without intermediaries.',
            'structured_output': StructuredOutput(
                area='DeFi Education',
                sources='Compound protocol docs, Aave whitepaper, DeFi Pulse data',
                relevance='Understanding DeFi lending is crucial for yield generation',
                message='DeFi lending works through smart contracts that automatically match lenders and borrowers. Users can earn yield by providing liquidity to lending pools.',
                anticipated_engagement='800 likes, 150 retweets',
                confidence=0.78,
                evidence_quality=0.85
            ),
            'unstructured_output': 'DeFi lending lets you earn interest on your crypto.',
            'actual_engagement': {'likes': 765, 'retweets': 142}
        }
    ] 