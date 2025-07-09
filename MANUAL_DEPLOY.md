# Manual H200 Deployment Guide

This guide provides the exact steps to deploy the Xinfluencer AI system to the H200 server.

### Step 1: Connect to the H200 Server

```bash
ssh -i /Users/max/Xinfluencer/influencer.pem ubuntu@157.10.162.127
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/your-repo/xinfluencer.git /home/ubuntu/xinfluencer
cd /home/ubuntu/xinfluencer
```

### Step 3: Create and Activate the Virtual Environment

```bash
sudo apt-get update
sudo apt-get install -y python3-virtualenv
virtualenv xinfluencer_env
source xinfluencer_env/bin/activate
```

### Step 4: Install Dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Step 5: Install H200-Specific Packages

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install flash-attn --no-build-isolation
```

### Step 6: Run the Test Suite

```bash
python3 scripts/test_h200_setup.py
``` 