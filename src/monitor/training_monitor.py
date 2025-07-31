#!/usr/bin/env python3
"""Real-time training monitor for H200 system impact."""

import subprocess
import time
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Real-time monitor for training system impact."""
    
    def __init__(self, ssh_key_path: str = "influencer.pem", host: str = "157.10.162.127"):
        self.ssh_key_path = ssh_key_path
        self.host = host
        self.monitoring = False
        self.monitor_thread = None
        self.history = []
        
    def run_ssh_command(self, command: str) -> tuple[str, str, int]:
        """Run SSH command and return output."""
        try:
            result = subprocess.run([
                'ssh', '-i', self.ssh_key_path, 
                '-o', 'StrictHostKeyChecking=no',
                f'ubuntu@{self.host}', command
            ], capture_output=True, text=True, timeout=30)
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "", "Timeout", 1
        except Exception as e:
            return "", str(e), 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {}
        
        # Uptime and load
        uptime, _, code = self.run_ssh_command("uptime")
        if code == 0:
            status['uptime'] = uptime
        
        # Memory usage
        memory, _, code = self.run_ssh_command("free -h")
        if code == 0:
            lines = memory.split('\n')
            if len(lines) > 1:
                mem_line = lines[1].split()
                if len(mem_line) >= 7:
                    status['memory'] = {
                        'total': mem_line[1],
                        'used': mem_line[2],
                        'free': mem_line[3],
                        'available': mem_line[6]
                    }
        
        # Disk usage
        disk, _, code = self.run_ssh_command("df -h /home/ubuntu/xinfluencer")
        if code == 0:
            lines = disk.split('\n')
            if len(lines) > 1:
                disk_line = lines[1].split()
                if len(disk_line) >= 5:
                    status['disk'] = {
                        'used': disk_line[2],
                        'available': disk_line[3],
                        'usage_percent': disk_line[4]
                    }
        
        return status
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get detailed GPU status."""
        gpu_info, _, code = self.run_ssh_command(
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits"
        )
        if code == 0 and gpu_info:
            try:
                name, util, mem_used, mem_total, temp, power = gpu_info.split(', ')
                return {
                    'name': name,
                    'utilization': int(util),
                    'memory_used': int(mem_used),
                    'memory_total': int(mem_total),
                    'memory_percent': round(int(mem_used) / int(mem_total) * 100, 1),
                    'temperature': int(temp),
                    'power_draw': float(power) if power != 'N/A' else None
                }
            except (ValueError, IndexError):
                pass
        return {}
    
    def get_training_processes(self) -> list:
        """Get training-related processes."""
        processes, _, code = self.run_ssh_command(
            "ps aux | grep python3 | grep -v grep | grep -E '(training|lora|run_training)'"
        )
        if code == 0 and processes:
            return [line.strip() for line in processes.split('\n') if line.strip()]
        return []
    
    def get_training_checkpoints(self) -> Dict[str, Any]:
        """Get training checkpoint status."""
        checkpoints, _, code = self.run_ssh_command(
            "cd /home/ubuntu/xinfluencer && find lora_checkpoints/ -name 'final_adapter' -type d 2>/dev/null | wc -l"
        )
        if code == 0:
            try:
                count = int(checkpoints.strip())
                return {'adapter_count': count}
            except ValueError:
                pass
        
        # Get latest checkpoint info
        latest, _, code = self.run_ssh_command(
            "cd /home/ubuntu/xinfluencer && ls -la lora_checkpoints/*/final_adapter/adapter_model.safetensors 2>/dev/null | tail -1"
        )
        if code == 0 and latest:
            return {'latest_checkpoint': latest.strip()}
        
        return {}
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get complete system snapshot."""
        timestamp = datetime.now()
        snapshot = {
            'timestamp': timestamp.isoformat(),
            'system': self.get_system_status(),
            'gpu': self.get_gpu_status(),
            'processes': self.get_training_processes(),
            'checkpoints': self.get_training_checkpoints()
        }
        
        # Store in history (keep last 100 snapshots)
        self.history.append(snapshot)
        if len(self.history) > 100:
            self.history.pop(0)
        
        return snapshot
    
    def print_snapshot(self, snapshot: Dict[str, Any], show_history: bool = True):
        """Print formatted snapshot."""
        timestamp = datetime.fromisoformat(snapshot['timestamp'])
        
        print(f"\n{'='*60}")
        print(f"H200 Training Monitor - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # System Status
        if snapshot['system']:
            print("\n📊 SYSTEM STATUS:")
            if 'uptime' in snapshot['system']:
                print(f"  Uptime: {snapshot['system']['uptime']}")
            if 'memory' in snapshot['system']:
                mem = snapshot['system']['memory']
                print(f"  Memory: {mem['used']}/{mem['total']} ({mem['available']} available)")
            if 'disk' in snapshot['system']:
                disk = snapshot['system']['disk']
                print(f"  Disk: {disk['used']}/{disk['available']} ({disk['usage_percent']} used)")
        
        # GPU Status
        if snapshot['gpu']:
            gpu = snapshot['gpu']
            print(f"\n🚀 GPU STATUS:")
            print(f"  GPU: {gpu.get('name', 'Unknown')}")
            print(f"  Utilization: {gpu.get('utilization', 0)}%")
            print(f"  Memory: {gpu.get('memory_used', 0)}MB / {gpu.get('memory_total', 0)}MB ({gpu.get('memory_percent', 0)}%)")
            print(f"  Temperature: {gpu.get('temperature', 0)}°C")
            if gpu.get('power_draw'):
                print(f"  Power: {gpu['power_draw']}W")
        
        # Training Processes
        if snapshot['processes']:
            print(f"\n⚙️  TRAINING PROCESSES ({len(snapshot['processes'])}):")
            for proc in snapshot['processes'][:3]:  # Show first 3
                print(f"  {proc}")
            if len(snapshot['processes']) > 3:
                print(f"  ... and {len(snapshot['processes']) - 3} more")
        else:
            print(f"\n⚙️  TRAINING PROCESSES: None active")
        
        # Checkpoints
        if snapshot['checkpoints']:
            print(f"\n💾 CHECKPOINTS:")
            for key, value in snapshot['checkpoints'].items():
                print(f"  {key}: {value}")
        
        # Historical Impact
        if show_history and len(self.history) > 1:
            print(f"\n📈 HISTORICAL IMPACT:")
            first = self.history[0]
            last = self.history[-1]
            
            if first['gpu'] and last['gpu']:
                gpu_util_change = last['gpu'].get('utilization', 0) - first['gpu'].get('utilization', 0)
                gpu_mem_change = last['gpu'].get('memory_used', 0) - first['gpu'].get('memory_used', 0)
                print(f"  GPU Utilization Change: {gpu_util_change:+d}%")
                print(f"  GPU Memory Change: {gpu_mem_change:+d}MB")
            
            time_diff = timestamp - datetime.fromisoformat(first['timestamp'])
            print(f"  Monitoring Duration: {time_diff}")
        
        print(f"\n{'='*60}")
    
    def start_monitoring(self, interval: int = 30):
        """Start background monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    snapshot = self.get_snapshot()
                    self.print_snapshot(snapshot)
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started background monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped background monitoring")
    
    def get_training_impact_summary(self) -> Dict[str, Any]:
        """Get summary of training impact from history."""
        if len(self.history) < 2:
            return {}
        
        first = self.history[0]
        last = self.history[-1]
        
        summary = {
            'monitoring_duration': str(datetime.fromisoformat(last['timestamp']) - datetime.fromisoformat(first['timestamp'])),
            'snapshots_taken': len(self.history),
            'peak_gpu_utilization': max(s.get('gpu', {}).get('utilization', 0) for s in self.history),
            'peak_gpu_memory': max(s.get('gpu', {}).get('memory_used', 0) for s in self.history),
            'peak_temperature': max(s.get('gpu', {}).get('temperature', 0) for s in self.history),
            'process_count_range': {
                'min': min(len(s.get('processes', [])) for s in self.history),
                'max': max(len(s.get('processes', [])) for s in self.history)
            }
        }
        
        return summary 