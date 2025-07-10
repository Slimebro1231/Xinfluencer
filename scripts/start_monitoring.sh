#!/bin/bash

# Xinfluencer AI Monitoring Dashboard Startup Script
# This script starts the comprehensive monitoring dashboard

set -e

# Configuration
DASHBOARD_PORT=8000
DASHBOARD_HOST="0.0.0.0"
LOG_FILE="/home/ubuntu/xinfluencer/logs/monitoring.log"
PID_FILE="/home/ubuntu/xinfluencer/monitoring.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        print_warning "Monitoring dashboard already running with PID $PID"
        print_status "Dashboard available at: http://localhost:$DASHBOARD_PORT"
        exit 0
    else
        print_warning "Stale PID file found, removing..."
        rm -f "$PID_FILE"
    fi
fi

# Create logs directory
mkdir -p "$(dirname "$LOG_FILE")"

print_status "Starting Xinfluencer AI Monitoring Dashboard..."

# Activate virtual environment
cd /home/ubuntu/xinfluencer
source xinfluencer_env/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/home/ubuntu/xinfluencer/models
export HF_HOME=/home/ubuntu/xinfluencer/models

# Check GPU status
print_status "Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    GPU_STATUS=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "GPU_ERROR")
    if [[ "$GPU_STATUS" != "GPU_ERROR" ]]; then
        print_success "GPU Status: $GPU_STATUS"
    else
        print_warning "Could not get GPU status"
    fi
else
    print_warning "nvidia-smi not available"
fi

# Start the monitoring dashboard
print_status "Starting dashboard on $DASHBOARD_HOST:$DASHBOARD_PORT..."

nohup python3 -m src.monitor.dashboard > "$LOG_FILE" 2>&1 &
DASHBOARD_PID=$!

# Save PID
echo $DASHBOARD_PID > "$PID_FILE"

# Wait a moment for startup
sleep 3

# Check if dashboard started successfully
if ps -p "$DASHBOARD_PID" > /dev/null 2>&1; then
    print_success "Monitoring dashboard started successfully!"
    print_status "PID: $DASHBOARD_PID"
    print_status "Log file: $LOG_FILE"
    print_status "Dashboard URL: http://localhost:$DASHBOARD_PORT"
    print_status "Metrics endpoint: http://localhost:$DASHBOARD_PORT/metrics"
    print_status "Health check: http://localhost:$DASHBOARD_PORT/api/health"
    echo ""
    print_status "Available endpoints:"
    echo "  - Dashboard: http://localhost:$DASHBOARD_PORT/"
    echo "  - Prometheus metrics: http://localhost:$DASHBOARD_PORT/metrics"
    echo "  - Quality metrics: http://localhost:$DASHBOARD_PORT/api/metrics/quality"
    echo "  - Retrieval metrics: http://localhost:$DASHBOARD_PORT/api/metrics/retrieval"
    echo "  - Generation metrics: http://localhost:$DASHBOARD_PORT/api/metrics/generation"
    echo "  - Behavioral metrics: http://localhost:$DASHBOARD_PORT/api/metrics/behavioral"
    echo "  - System health: http://localhost:$DASHBOARD_PORT/api/health"
    echo "  - Active alerts: http://localhost:$DASHBOARD_PORT/api/alerts"
    echo "  - WebSocket: ws://localhost:$DASHBOARD_PORT/ws"
    echo ""
    print_success "🎉 Monitoring dashboard is ready!"
else
    print_error "Failed to start monitoring dashboard"
    print_status "Check logs: tail -f $LOG_FILE"
    exit 1
fi 