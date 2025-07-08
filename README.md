# Xinfluencer AI

A self-learning AI agent that analyzes crypto influencer content and generates insights.

## Architecture

The system implements a bot-influencer architecture with a data flywheel:

1. **Data Ingestion**: Fetches tweets from ~100 trusted KOL accounts
2. **Quality Gate**: Automated filters for language, toxicity, bot detection, and perplexity
3. **Vector Storage**: Clean chunks stored in GPU-backed vector database (Qdrant)
4. **Self-RAG Generation**: Retrieve → draft → re-retrieve & critique approach
5. **Multi-layered Review**: Human, AI peer (GPT-4o), and Twitter engagement feedback
6. **Continuous Learning**: PPO policy updates weekly + daily LoRA micro-tuning

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main pipeline:
```bash
python src/main.py
```

## Project Structure

- `src/`: Main source code
  - `data/`: Data ingestion, filtering, and chunking
  - `vector/`: Vector database operations
  - `model/`: LLM generation and LoRA fine-tuning
  - `review/`: Human and AI review systems
  - `monitor/`: Dashboard and RAGAS evaluation
- `tests/`: Unit tests
- `scripts/`: Setup and utility scripts

## Key Features

- **Self-RAG**: Model critiques and iterates before final output
- **LoRA Fine-tuning**: Daily parameter-efficient model updates
- **Multi-modal Feedback**: Human, AI, and engagement-based scoring
- **Real-time Monitoring**: Prometheus + Grafana dashboard with RAGAS metrics

For detailed architecture, see [flow.md](flow.md). 