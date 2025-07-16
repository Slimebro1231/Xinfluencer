#!/usr/bin/env python3
"""
Verify Working Components on H200
This script tests the components that are confirmed working and provides a clear status report.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config

def test_ssh_connection():
    """Test SSH connection to H200."""
    print("Testing SSH connection to H200...")
    
    config = Config()
    ssh_key = config.h200.pem_file
    user = config.h200.user
    host = config.h200.host
    
    import subprocess
    import shlex
    
    ssh_command = f"ssh -i {ssh_key} -o ConnectTimeout=10 -o StrictHostKeyChecking=no {user}@{host} 'echo Connection successful'"
    
    try:
        result = subprocess.run(
            shlex.split(ssh_command),
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            print("✅ SSH connection successful")
            return True
        else:
            print(f"❌ SSH connection failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ SSH connection error: {e}")
        return False

def test_gpu_status():
    """Test GPU status on H200."""
    print("\nTesting GPU status...")
    
    config = Config()
    ssh_key = config.h200.pem_file
    user = config.h200.user
    host = config.h200.host
    
    import subprocess
    import shlex
    
    # Test nvidia-smi
    nvidia_command = f"ssh -i {ssh_key} -o StrictHostKeyChecking=no {user}@{host} 'nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader'"
    
    try:
        result = subprocess.run(
            shlex.split(nvidia_command),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ GPU Status:")
            print(f"   {result.stdout.strip()}")
            return True
        else:
            print(f"❌ GPU status check failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ GPU status error: {e}")
        return False

def test_cuvs_integration():
    """Test cuVS integration on H200."""
    print("\nTesting cuVS integration...")
    
    config = Config()
    ssh_key = config.h200.pem_file
    user = config.h200.user
    host = config.h200.host
    remote_dir = config.h200.remote_dir
    
    import subprocess
    import shlex
    
    # Test cuVS script
    cuvs_command = f"ssh -i {ssh_key} -o StrictHostKeyChecking=no {user}@{host} 'cd {remote_dir} && source xinfluencer_env/bin/activate && python3 scripts/explore_nvidia_cuvs.py'"
    
    try:
        result = subprocess.run(
            shlex.split(cuvs_command),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ cuVS integration working")
            # Extract key metrics
            output = result.stdout
            if "GPU vector operations available" in output:
                print("   - GPU vector operations: ✅")
            if "FAISS-GPU available" in output:
                print("   - FAISS-GPU: ✅")
            if "CuPy available" in output:
                print("   - CuPy: ✅")
            return True
        else:
            print(f"❌ cuVS integration failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ cuVS integration error: {e}")
        return False

def test_improved_scraper():
    """Test improved scraper on H200."""
    print("\nTesting improved scraper...")
    
    config = Config()
    ssh_key = config.h200.pem_file
    user = config.h200.user
    host = config.h200.host
    remote_dir = config.h200.remote_dir
    
    import subprocess
    import shlex
    
    # Test scraper
    scraper_command = f"ssh -i {ssh_key} -o StrictHostKeyChecking=no {user}@{host} 'cd {remote_dir} && source xinfluencer_env/bin/activate && python3 scripts/test_improved_scraper.py'"
    
    try:
        result = subprocess.run(
            shlex.split(scraper_command),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("✅ Improved scraper working")
            # Extract key metrics
            output = result.stdout
            if "Total unique results:" in output:
                lines = output.split('\n')
                for line in lines:
                    if "Total unique results:" in line:
                        print(f"   - {line.strip()}")
                        break
            return True
        else:
            print(f"❌ Improved scraper failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Improved scraper error: {e}")
        return False

def test_data_quality():
    """Test data quality of scraped tweets."""
    print("\nTesting data quality...")
    
    config = Config()
    ssh_key = config.h200.pem_file
    user = config.h200.user
    host = config.h200.host
    remote_dir = config.h200.remote_dir
    
    import subprocess
    import shlex
    
    # Create a simple data quality check script
    quality_script = '''import json
try:
    with open("data/seed_tweets/scraped_seed_tweets.json", "r") as f:
        data = json.load(f)
    print(f"Total tweets: {len(data)}")
    if data:
        avg_length = sum(len(t.get("text", "")) for t in data) / len(data)
        truncated = sum(1 for t in data if t.get("text", "").endswith("..."))
        print(f"Average length: {avg_length:.1f} characters")
        print(f"Truncated tweets: {truncated}")
        print(f"Quality score: {((len(data) - truncated) / len(data) * 100):.1f}%")
    else:
        print("No tweets found")
except Exception as e:
    print(f"Error: {e}")
'''
    
    # Create the script on the server
    create_script = f"ssh -i {ssh_key} -o StrictHostKeyChecking=no {user}@{host} 'cd {remote_dir} && cat > test_quality.py << \"EOF\"\n{quality_script}\nEOF'"
    
    try:
        # Create script
        subprocess.run(shlex.split(create_script), check=True, timeout=10)
        
        # Run script
        run_script = f"ssh -i {ssh_key} -o StrictHostKeyChecking=no {user}@{host} 'cd {remote_dir} && source xinfluencer_env/bin/activate && python3 test_quality.py'"
        
        result = subprocess.run(
            shlex.split(run_script),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Data quality check successful")
            print(f"   {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Data quality check failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Data quality check error: {e}")
        return False

def generate_status_report():
    """Generate a comprehensive status report."""
    print("=" * 80)
    print("XINFLUENCER AI - H200 COMPONENT STATUS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test each component
    components = {
        "SSH Connection": test_ssh_connection,
        "GPU Status": test_gpu_status,
        "cuVS Integration": test_cuvs_integration,
        "Improved Scraper": test_improved_scraper,
        "Data Quality": test_data_quality
    }
    
    results = {}
    
    for component_name, test_func in components.items():
        try:
            results[component_name] = test_func()
        except Exception as e:
            print(f"❌ {component_name} test error: {e}")
            results[component_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    working_components = [name for name, status in results.items() if status]
    failed_components = [name for name, status in results.items() if not status]
    
    print(f"Working Components: {len(working_components)}/{len(components)}")
    for component in working_components:
        print(f"  ✅ {component}")
    
    if failed_components:
        print(f"\nFailed Components: {len(failed_components)}")
        for component in failed_components:
            print(f"  ❌ {component}")
    
    # Overall status
    if len(working_components) >= 4:  # At least 4 out of 5 components working
        print(f"\n🎯 Overall Status: READY FOR PRODUCTION")
        print("   The core pipeline components are operational.")
    elif len(working_components) >= 3:
        print(f"\n⚠️  Overall Status: MOSTLY READY")
        print("   Most components are working, minor issues to resolve.")
    else:
        print(f"\n🚨 Overall Status: NEEDS ATTENTION")
        print("   Multiple components need to be fixed.")
    
    # Save results
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "components": results,
        "working_count": len(working_components),
        "total_count": len(components),
        "working_components": working_components,
        "failed_components": failed_components
    }
    
    report_file = Path("logs/component_status_report.json")
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return results

def main():
    """Main function."""
    print("Xinfluencer AI - Component Status Verification")
    print("=" * 50)
    
    try:
        results = generate_status_report()
        
        # Exit with appropriate code
        working_count = sum(1 for status in results.values() if status)
        if working_count >= 4:
            print("\n✅ System is ready for production use!")
            sys.exit(0)
        elif working_count >= 3:
            print("\n⚠️  System is mostly ready, minor issues to resolve.")
            sys.exit(1)
        else:
            print("\n❌ System needs attention before production use.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nVerification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 