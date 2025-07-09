# H200 AI Quickstart Guide

This guide provides a quick overview of how to set up and use the Xinfluencer AI system on an H200 GPU.

## 1. Prerequisites

- Access to an H200 server with NVIDIA drivers installed.
- An SSH key with access to the server.

## 2. Setup

Follow the instructions in `MANUAL_DEPLOY.md` to set up the environment on the H200 server.

## 3. Using the CLI

The primary way to interact with the AI is through the command-line interface (`src/cli.py`).

### Generate Text

To generate text from a prompt:

```bash
# Activate the virtual environment
source xinfluencer_env/bin/activate

# Run the CLI
python3 src/cli.py generate --prompt "What are the latest trends in DeFi?"
```

### Interactive Mode

For a chat-like experience, use the interactive mode:

```bash
# Activate the virtual environment
source xinfluencer_env/bin/activate

# Run the CLI
python3 src/cli.py interactive
```

### Check Status

To check the model and GPU status:

```bash
# Activate the virtual environment
source xinfluencer_env/bin/activate

# Run the CLI
python3 src/cli.py status
```

## 4. Running Tests

To verify the setup, run the test suite:

```bash
# Activate the virtual environment
source xinfluencer_env/bin/activate

# Run the tests
python3 scripts/test_h200_setup.py
```