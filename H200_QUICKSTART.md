# 🚀 H200 Quick Start Guide

This guide will help you set up and run the Xinfluencer AI system on the H200 GPU server.

## 📋 Prerequisites

- SSH access to H200 server (`157.10.162.127`)
- SSH key configured (`~/.ssh/id_rsa`)
- Basic familiarity with command line

## 🎯 Quick Start (3 Steps)

### Step 1: Deploy to H200
```bash
# Deploy the complete system to H200
./scripts/deploy_h200.sh
```

This script will:
- ✅ Test H200 connection
- ✅ Check GPU status
- ✅ Sync project files
- ✅ Install dependencies
- ✅ Set up environment
- ✅ Create startup scripts

### Step 2: SSH to H200
```bash
# Connect to H200 server
./ssh_h200.sh
```

### Step 3: Start the AI System
```bash
# On H200 server
cd /home/ubuntu/xinfluencer
./start_xinfluencer.sh
```

## 🤖 Using the AI System

### Command Line Interface (CLI)

The system includes a comprehensive CLI for easy interaction:

```bash
# Generate a simple response
python3 src/cli.py generate "What's the latest trend in crypto?"

# Generate with Self-RAG (recommended)
python3 src/cli.py rag "How should I invest in Bitcoin?"

# Review a response
python3 src/cli.py review "What's Bitcoin?" "Bitcoin is a cryptocurrency"

# Check GPU memory
python3 src/cli.py memory

# Interactive mode (recommended for testing)
python3 src/cli.py interactive
```

### Interactive Mode Examples

```bash
# Start interactive mode
python3 src/cli.py interactive

# In interactive mode:
🤖 Xinfluencer AI > What's the latest trend in crypto?
🤖 [AI response will appear here]

🤖 Xinfluencer AI > rag How should I invest in Bitcoin?
🧠 Response: [Self-RAG response with context]
📊 Score: 8.5

🤖 Xinfluencer AI > memory
💾 GPU Memory: 12.5GB / 80.0GB

🤖 Xinfluencer AI > quit
👋 Goodbye!
```

## 🔧 System Management

### Monitor System Status
```bash
# Check GPU and system status
./monitor_h200.sh
```

### Test H200 Setup
```bash
# Run comprehensive tests
python3 scripts/test_h200_setup.py
```

### Check Logs
```bash
# View system logs
tail -f logs/xinfluencer.log
```

## 🎛️ Configuration

### Model Configuration
The system uses **Mistral-7B** by default with 4-bit quantization for H200 optimization.

Key settings in `src/config.py`:
- `generation_model`: "mistralai/Mistral-7B-v0.1"
- `embedding_model`: "sentence-transformers/all-MiniLM-L6-v2"
- Quantization: Enabled for memory efficiency

### Environment Variables
Create `.env` file on H200 server:
```bash
# Twitter API (optional)
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_BEARER_TOKEN=your_bearer_token

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Model Settings
GENERATION_MODEL=mistralai/Mistral-7B-v0.1
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Pipeline Settings
MAX_TWEETS_PER_KOL=50
CHUNK_SIZE=256
CHUNK_OVERLAP=50
```

## 📊 Performance Expectations

### H200 Performance
- **Model Loading**: ~2-3 minutes (first time)
- **Generation Speed**: ~50-100 tokens/second
- **Memory Usage**: ~12-15GB with 4-bit quantization
- **Concurrent Requests**: 1-2 (recommended)

### Model Capabilities
- **Context Length**: 8K tokens
- **Generation Quality**: High (Mistral-7B)
- **Self-RAG**: Enabled for factual accuracy
- **Review System**: Multi-criteria evaluation

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Clear GPU memory
python3 src/cli.py memory --clear

# Or restart the system
./start_xinfluencer.sh
```

**2. Model Loading Fails**
```bash
# Check internet connection
ping huggingface.co

# Clear model cache
rm -rf models/
```

**3. Slow Generation**
```bash
# Check GPU utilization
nvidia-smi

# Monitor memory usage
python3 src/cli.py memory
```

### Getting Help

**System Status Check:**
```bash
python3 src/cli.py status
```

**Full System Test:**
```bash
python3 scripts/test_h200_setup.py
```

**View Logs:**
```bash
tail -f logs/xinfluencer.log
```

## 🎯 Production Usage

### Recommended Workflow

1. **Start the system:**
   ```bash
   cd /home/ubuntu/xinfluencer
   ./start_xinfluencer.sh
   ```

2. **Use CLI for interactions:**
   ```bash
   python3 src/cli.py interactive
   ```

3. **Monitor performance:**
   ```bash
   ./monitor_h200.sh
   ```

4. **Check logs regularly:**
   ```bash
   tail -f logs/xinfluencer.log
   ```

### Best Practices

- ✅ Use Self-RAG for factual queries
- ✅ Monitor GPU memory usage
- ✅ Keep system logs for debugging
- ✅ Use interactive mode for testing
- ✅ Clear memory when needed

## 🚀 Advanced Features

### Self-RAG Generation
The system includes Self-RAG for improved factual accuracy:
```bash
python3 src/cli.py rag "What are the latest developments in DeFi?"
```

### Response Review
Automated review system evaluates responses:
```bash
python3 src/cli.py review "What's Bitcoin?" "Bitcoin is a cryptocurrency" "Context about Bitcoin..."
```

### Memory Management
Monitor and manage GPU memory:
```bash
python3 src/cli.py memory
python3 src/cli.py memory --clear
```

## 📈 Next Steps

1. **Customize the model** for your specific use case
2. **Integrate with Twitter API** for real data
3. **Set up monitoring dashboard** with Prometheus/Grafana
4. **Implement PPO training** for continuous improvement
5. **Add human review interface** for quality control

---

**🎉 You're now ready to use the AI system on H200!**

For questions or issues, check the logs or run the test suite.