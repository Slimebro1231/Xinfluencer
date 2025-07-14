# H200 Mistral Deployment Guide - Fixed Version

## Issues Addressed

The previous deployment attempts had several issues that have been fixed:

1. **Virtual Environment**: No isolation led to dependency conflicts
2. **Version Conflicts**: PyTorch installation conflicted with requirements.txt
3. **Memory Issues**: Insufficient CUDA memory management
4. **Error Handling**: Poor fallback mechanisms

## New Deployment Approach

### Step 1: Run the Fixed Deployment Script

```bash
./deploy_h200_fixed.sh
```

This improved script will:
- Create proper virtual environment isolation
- Install compatible dependencies step-by-step
- Use H200-optimized memory settings
- Include robust error handling and fallbacks
- Test each component before proceeding

### Step 2: SSH to H200 and Test

```bash
# SSH to H200 (script will provide the exact command)
ssh -i ~/.ssh/id_rsa ubuntu@157.10.162.127

# Navigate to project directory
cd /home/ubuntu/xinfluencer

# Test the deployment
./start_mistral_h200.sh
```

### Step 3: Run Model Tests

```bash
# Run comprehensive model test
python3 test_mistral.py

# Use the optimized CLI
./mistral_cli.sh interactive
```

## Key Improvements

### Virtual Environment Isolation
- Creates dedicated `xinfluencer_env` virtual environment
- Prevents system-wide dependency conflicts
- Checks if venv exists before creating (no re-creation)

### Dependency Management
- Uses streamlined `requirements_h200.txt`
- Installs PyTorch with correct CUDA version first
- Verifies each installation step

### Memory Optimization
- Conservative memory limits for H200
- Proper CUDA cache management
- Automatic fallback to smaller models if needed

### Error Recovery
- Graceful handling of model loading failures
- Fallback to DialoGPT-medium if Mistral fails
- Comprehensive error reporting

## Expected Results

After successful deployment:
- **Model Loading**: ~2-3 minutes for Mistral-7B
- **Memory Usage**: ~12-15GB with 4-bit quantization
- **Generation Speed**: ~30-60 tokens/second
- **Success Rate**: >95% with fallback mechanisms

## Troubleshooting

### If Deployment Fails:
1. Check SSH connectivity to H200
2. Verify GPU is available (`nvidia-smi`)
3. Ensure sufficient disk space (>50GB)
4. Check internet connectivity for model downloads

### If Model Loading Fails:
- The script will automatically fall back to DialoGPT-medium
- Check GPU memory availability
- Try with smaller batch sizes

### Common Commands:
```bash
# Check system status
./mistral_cli.sh status

# Test model health
python3 -c "
from src.model.generate_h200 import H200TextGenerator
gen = H200TextGenerator()
print(gen.health_check())
"

# Clear GPU memory
./mistral_cli.sh memory --clear
```

## Next Steps

Once basic deployment works:
1. Test with your specific use cases
2. Adjust memory settings if needed
3. Configure any additional integrations
4. Set up monitoring and logging

The new deployment approach prioritizes reliability over speed, ensuring a working system before optimization.