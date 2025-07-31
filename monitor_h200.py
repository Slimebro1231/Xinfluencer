#!/usr/bin/env python3
"""Simple H200 monitoring script for training progress."""

import subprocess
import time
import json
from datetime import datetime

def run_ssh_command(command):
    """Run SSH command and return output."""
    try:
        result = subprocess.run([
            'ssh', '-i', 'influencer.pem', 
            '-o', 'StrictHostKeyChecking=no',
            'ubuntu@157.10.162.127', command
        ], capture_output=True, text=True, timeout=30)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", 1
    except Exception as e:
        return "", str(e), 1

def get_system_status():
    """Get basic system status."""
    uptime, _, code = run_ssh_command("uptime")
    if code == 0:
        return uptime
    return "Connection failed"

def get_gpu_status():
    """Get GPU status."""
    gpu_info, _, code = run_ssh_command(
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
    )
    if code == 0 and gpu_info:
        return gpu_info
    return "GPU info unavailable"

def get_training_processes():
    """Get training-related processes."""
    processes, _, code = run_ssh_command(
        "ps aux | grep python3 | grep -v grep | grep -E '(training|lora|run_training)'"
    )
    if code == 0:
        return processes if processes else "No training processes found"
    return "Process info unavailable"

def get_training_checkpoints():
    """Get training checkpoint status."""
    checkpoints, _, code = run_ssh_command(
        "cd /home/ubuntu/xinfluencer && ls -la lora_checkpoints/ 2>/dev/null || echo 'No checkpoints'"
    )
    if code == 0:
        return checkpoints
    return "Checkpoint info unavailable"

def monitor_loop():
    """Main monitoring loop."""
    print("H200 Training Monitor")
    print("=" * 50)
    
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}]")
        print("-" * 30)
        
        # System status
        print("System Status:")
        print(f"  {get_system_status()}")
        
        # GPU status
        print("\nGPU Status:")
        gpu_status = get_gpu_status()
        if gpu_status != "GPU info unavailable":
            name, util, mem_used, mem_total, temp = gpu_status.split(', ')
            print(f"  GPU: {name}")
            print(f"  Utilization: {util}%")
            print(f"  Memory: {mem_used}MB / {mem_total}MB")
            print(f"  Temperature: {temp}°C")
        else:
            print(f"  {gpu_status}")
        
        # Training processes
        print("\nTraining Processes:")
        processes = get_training_processes()
        if processes != "Process info unavailable":
            for line in processes.split('\n'):
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"  {processes}")
        
        # Checkpoints
        print("\nTraining Checkpoints:")
        checkpoints = get_training_checkpoints()
        if "No checkpoints" not in checkpoints and "Checkpoint info unavailable" not in checkpoints:
            lines = checkpoints.split('\n')
            for line in lines[-5:]:  # Show last 5 lines
                if line.strip() and 'total' not in line:
                    print(f"  {line}")
        else:
            print(f"  {checkpoints}")
        
        print("\n" + "=" * 50)
        print("Press Ctrl+C to stop monitoring")
        
        # Wait 30 seconds before next check
        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    except Exception as e:
        print(f"Error: {e}") 