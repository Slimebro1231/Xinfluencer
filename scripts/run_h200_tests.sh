#!/bin/bash

# Comprehensive H200 Test Runner for Xinfluencer AI
# Runs all tests to verify H200 access and Twitter API functionality

set -e  # Exit on any error

echo "H200 Comprehensive Test Suite for Xinfluencer AI"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if we're in the right directory
check_environment() {
    print_status "Checking environment..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found. Please run from project root."
        exit 1
    fi
    
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating from template..."
        cp env.example .env
        print_warning "Please update .env with your Twitter API credentials before running tests."
    fi
    
    print_success "Environment check passed"
}

# Install dependencies if needed
install_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if virtual environment exists
    if [ ! -d "xinfluencer_env" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv xinfluencer_env
    fi
    
    # Activate virtual environment
    source xinfluencer_env/bin/activate
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Test 1: Basic H200 Access
test_h200_access() {
    print_status "Test 1: H200 GPU Access Test"
    echo "--------------------------------"
    
    if python3 scripts/test_h200_access.py; then
        print_success "H200 access test completed"
        return 0
    else
        print_error "H200 access test failed"
        return 1
    fi
}

# Test 2: Twitter API Access
test_twitter_api() {
    print_status "Test 2: Twitter API Access Test"
    echo "------------------------------------"
    
    if python3 scripts/test_twitter_api.py; then
        print_success "Twitter API test completed"
        return 0
    else
        print_error "Twitter API test failed"
        return 1
    fi
}

# Test 3: Full Performance Suite
test_performance_suite() {
    print_status "Test 3: Full H200 Performance Suite"
    echo "----------------------------------------"
    
    if python3 tests/test_h200_performance.py; then
        print_success "Performance suite completed"
        return 0
    else
        print_error "Performance suite failed"
        return 1
    fi
}

# Test 4: Basic Pipeline Test
test_basic_pipeline() {
    print_status "Test 4: Basic Pipeline Test"
    echo "-------------------------------"
    
    # Test the basic data ingestion pipeline
    if python3 -c "
import sys
sys.path.append('src')
from data.ingest import TwitterIngester
from data.filter import DataFilter
from data.chunk import TextChunker
from config import Config

config = Config()
print('✅ Configuration loaded successfully')

ingester = TwitterIngester(config)
print('✅ Twitter ingester initialized')

filter_obj = DataFilter(config)
print('✅ Data filter initialized')

chunker = TextChunker(config)
print('✅ Text chunker initialized')

print('✅ Basic pipeline components working')
"; then
        print_success "Basic pipeline test completed"
        return 0
    else
        print_error "Basic pipeline test failed"
        return 1
    fi
}

# Generate summary report
generate_summary() {
    print_status "Generating test summary..."
    
    echo ""
    echo "Test Summary Report"
    echo "====================="
    echo "Timestamp: $(date)"
    echo ""
    
    # Check for result files
    echo "Generated Files:"
    if ls h200_access_test_*.json 1> /dev/null 2>&1; then
        echo "SUCCESS: H200 access test results"
    else
        echo "ERROR: H200 access test results not found"
    fi
    
    if ls twitter_api_test_*.json 1> /dev/null 2>&1; then
        echo "SUCCESS: Twitter API test results"
    else
        echo "ERROR: Twitter API test results not found"
    fi
    
    if ls h200_performance_results_*.json 1> /dev/null 2>&1; then
        echo "SUCCESS: Performance test results"
    else
        echo "ERROR: Performance test results not found"
    fi
    
    if ls h200_performance_summary_*.txt 1> /dev/null 2>&1; then
        echo "SUCCESS: Performance summary"
    else
        echo "ERROR: Performance summary not found"
    fi
    
    echo ""
    echo "Next Steps:"
    echo "1. Review test results in the generated JSON files"
    echo "2. If all tests pass, run: ./scripts/deploy_h200.sh"
    echo "3. Start the agent with: ./start_xinfluencer.sh"
}

# Main test runner
main() {
    local test_results=()
    
    # Setup
    check_environment
    install_dependencies
    
    echo ""
    echo "Running Test Suite..."
    echo "========================"
    
    # Run tests
    if test_h200_access; then
        test_results+=("H200 Access: ✅ PASS")
    else
        test_results+=("H200 Access: ❌ FAIL")
    fi
    
    if test_twitter_api; then
        test_results+=("Twitter API: ✅ PASS")
    else
        test_results+=("Twitter API: ❌ FAIL")
    fi
    
    if test_performance_suite; then
        test_results+=("Performance Suite: ✅ PASS")
    else
        test_results+=("Performance Suite: ❌ FAIL")
    fi
    
    if test_basic_pipeline; then
        test_results+=("Basic Pipeline: ✅ PASS")
    else
        test_results+=("Basic Pipeline: ❌ FAIL")
    fi
    
    # Print results
    echo ""
    echo "📊 Test Results Summary"
    echo "======================"
    for result in "${test_results[@]}"; do
        echo "$result"
    done
    
    # Count failures
    local failures=$(echo "${test_results[@]}" | grep -o "❌ FAIL" | wc -l)
    
    if [ "$failures" -eq 0 ]; then
        echo ""
        print_success "All tests passed! H200 is ready for deployment."
        generate_summary
    else
        echo ""
        print_warning "$failures test(s) failed. Please review the results above."
        print_warning "Check the error messages and fix any issues before deployment."
    fi
}

# Run main function
main "$@" 