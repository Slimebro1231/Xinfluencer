# Xinfluencer AI

A self-learning AI agent that analyzes and generates crypto content using advanced retrieval-augmented generation, LoRA fine-tuning, and continuous learning from real engagement data.

## What Makes Us Unique

**Continuous Learning from Real Engagement**: Unlike traditional AI systems that rely on static training data, Xinfluencer learns continuously from actual social media engagement metrics. Every tweet's performance feeds back into the system, creating a data flywheel that constantly improves content quality.

**Self-Reflective AI**: Our agent doesn't just generate content—it critiques and improves its own work through Self-RAG (Retrieval-Augmented Generation). Before posting, the AI retrieves relevant context, drafts content, then re-retrieves and critiques its own output for accuracy and relevance.

**Multi-Signal Quality Control**: Content quality is evaluated through three independent channels: human review, AI peer review using raw GPT-4o, and real Twitter engagement metrics. This triangulation ensures consistent quality while maintaining authenticity.

**Domain-Specific Expertise**: Focused specifically on cryptocurrency and Real World Assets (RWA), the system builds deep expertise through curated knowledge from trusted Key Opinion Leaders (KOLs) in the space.

**LoRA-Powered Identity Training**: Our Soju AI personality is fine-tuned using LoRA (Low-Rank Adaptation) on high-quality crypto content, enabling progressive training that continuously improves the model's crypto expertise and personality.

## How It Works

### 1. Knowledge Acquisition

The system continuously monitors trusted crypto influencers, filtering content through automated quality gates that check for language, toxicity, bot activity, and engagement metrics. Only high-quality content is processed and stored in our vector database.

### 2. Intelligent Content Generation

When generating new content, the AI:

- Retrieves relevant context from our curated knowledge base
- Drafts initial content using advanced language models
- Re-retrieves supporting evidence and critiques its own work
- Iteratively improves the content before finalizing

### 3. Multi-Channel Evaluation

Every piece of content is evaluated through:

- **Human Review**: Expert oversight during early phases
- **AI Peer Review**: Automated critique using raw GPT-4o
- **Engagement Metrics**: Real Twitter performance data

### 4. Continuous Improvement

Performance data feeds into our reward model, which drives:

- **Progressive LoRA Training**: Continuous fine-tuning with checkpoint continuation
- **Daily Model Updates**: Incremental improvements to the Soju personality
- **Weekly Policy Updates**: Reinforcement learning from human feedback
- **Real-time Monitoring**: Quality drift detection and correction

## Architecture Overview

```
Data Flywheel
├── Knowledge Ingestion (KOL monitoring)
├── Quality Filtering (automated gates)
├── Vector Storage (semantic search)
├── LoRA Identity Training (Soju personality fine-tuning)
├── Self-RAG Generation (retrieve → draft → critique)
├── Multi-Signal Evaluation (human + AI + engagement)
└── Continuous Learning (progressive LoRA + PPO updates)
```

## Key Technologies

- **Self-RAG**: Self-reflective retrieval-augmented generation
- **LoRA**: Parameter-efficient fine-tuning for progressive identity training
- **Soju Generator**: Specialized crypto content generation with personality
- **PPO**: Reinforcement learning from human feedback
- **Vector Search**: Semantic similarity for context retrieval
- **Multi-Signal Evaluation**: Triangulated quality assessment

## Performance Metrics

Our system tracks comprehensive quality metrics including:

- **Retrieval Precision**: Context relevance and accuracy
- **Faithfulness**: Factual consistency with source material
- **Engagement Rate**: Real social media performance
- **Human Preference**: Expert evaluation scores
- **Brand Safety**: Toxicity and off-topic content detection

## LoRA Training Performance

Our LoRA-based identity training achieves:

- **Training Loss**: 0.76 (optimized for H200 GPU)
- **LoRA Parameters**: 13.6M with 0% sparsity
- **Training Time**: ~30 seconds per epoch on H200
- **Progressive Training**: Continues from existing checkpoints
- **Clear Differentiation**: LoRA outputs differ from base model

## Use Cases

**Content Creation**: Generate high-quality crypto content that resonates with target audiences
**Market Analysis**: Analyze trends and insights from trusted sources
**Engagement Optimization**: Continuously improve content performance through data-driven learning
**Brand Safety**: Maintain consistent quality and tone across all generated content

## Soju AI Personality

Our LoRA-trained Soju AI specializes in:

- **Professional Crypto Commentary**: Market analysis and insights
- **Educational Content**: Explaining complex crypto concepts
- **Trend Analysis**: Identifying and discussing market trends
- **Regulatory Updates**: Analysis of crypto regulation developments

### Content Types

- **Tweets**: Professional, casual, educational, and analytical styles
- **Threads**: Detailed explanations and analysis
- **Market Updates**: Real-time crypto market commentary
- **Educational Posts**: Beginner-friendly crypto explanations

## Getting Started

### LoRA Training

```bash
# Run progressive LoRA training on H200
python3 run_training_official.py
```

### Content Generation

```bash
# Generate Soju-style crypto tweets
python3 src/model/soju_generator.py --topic "Bitcoin adoption" --style professional

# Generate batch of tweets
python3 src/model/soju_generator.py --batch --output tweets.json

# Generate daily content package
python3 src/model/soju_generator.py --daily --output daily_content.json

# Use base model only (no LoRA)
python3 src/model/soju_generator.py --topic "DeFi" --no-lora
```

### Programmatic Usage

```python
from src.model.soju_generator import SojuGenerator

# Initialize with LoRA support
generator = SojuGenerator(use_lora=True)

# Generate content
tweet = generator.generate_tweet("Bitcoin price action", "professional")
analysis = generator.generate_content("analysis", "DeFi protocols")
```

For technical implementation details and deployment instructions, see our internal documentation.

---

**Status**: Production Ready with LoRA Integration
**Last Updated**: January 2025
