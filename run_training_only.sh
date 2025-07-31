#!/bin/bash

# Simple script to run LoRA training on H200 server

set -e

H200_HOST="157.10.162.127"
H200_USER="ubuntu"
PROJECT_DIR="/home/ubuntu/xinfluencer"
PEM_FILE="/Users/max/Xinfluencer/influencer.pem"

echo "Starting LoRA training on H200 server..."

# Test SSH connection
echo "Testing SSH connection to H200..."
if ! ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${H200_USER}@${H200_HOST} "echo 'SSH connection successful'"; then
    echo "ERROR: Cannot connect to H200 server"
    exit 1
fi

# Sync project files
echo "Syncing project files to H200..."
rsync -avz -e "ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no" --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='tests' \
    --exclude='*.log' --exclude='data/cache' \
    . \
    ${H200_USER}@${H200_HOST}:${PROJECT_DIR}/

# Run LoRA training
echo "Starting LoRA training on H200..."
ssh -i ${PEM_FILE} -o StrictHostKeyChecking=no ${H200_USER}@${H200_HOST} "cd ${PROJECT_DIR} && source xinfluencer_env/bin/activate && python3 run_training_remote.py"

echo "LoRA training completed!" 