#!/bin/bash
# Enterprise Test Automation Script for CC-Excellence
# Comprehensive testing pipeline with different test suites

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/workspaces/CC-Excellence"
TEST_DIR="$PROJECT_ROOT/tests"
REPORTS_DIR="$TEST_DIR/reports"

# Create reports directory
mkdir -p "$REPORTS_DIR"

echo -e "${BLUE}🚀 CC-Excellence Enterprise Test Suite${NC}"
echo "==============================================="

# Function to print test section headers
print_section() {
    echo -e "\n${YELLOW}📋 $1${NC}"
    echo "----------------------------------------"
}

# Function to run test suite with timing
run_test_suite() {
    local suite_name="$1"
    local pytest_args="$2"
    local start_time=$(date +%s)
    
    echo -e "${BLUE}Running $suite_name...${NC}"
    
    if pytest $pytest_args; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}✅ $suite_name completed successfully in ${duration}s${NC}"
        return 0
    else
        echo -e "${RED}❌ $suite_name failed${NC}"
        return 1
    fi
}

# Change to project root
cd "$PROJECT_ROOT"

print_section "Environment Setup"
echo "Project Root: $PROJECT_ROOT"
echo "Test Directory: $TEST_DIR"
echo "Python Version: $(python --version)"
echo "Pytest Version: $(pytest --version)"

# Check if required packages are installed
print_section "Dependency Check"
python -c "import pytest, pandas, numpy, plotly" && echo -e "${GREEN}✅ Core dependencies available${NC}" || {
    echo -e "${RED}❌ Missing core dependencies${NC}"
    echo "Installing test requirements..."
    pip install -r tests/requirements.txt
}

# Smoke Tests - Quick validation
print_section "Smoke Tests"
run_test_suite "Smoke Tests" "-m smoke --tb=short -q" || echo "No smoke tests defined yet"

# Unit Tests - Core business logic
print_section "Unit Tests"
run_test_suite "Unit Tests" "-m unit --cov=modules/prophet_core --cov=modules/prophet_presentation --cov-report=term-missing"

# Integration Tests - Module interactions  
print_section "Integration Tests"
run_test_suite "Integration Tests" "-m integration --cov=modules/prophet_module --cov-append"

# Prophet-specific test suites
print_section "Prophet Core Tests"
run_test_suite "Prophet Core" "-m prophet_core --tb=short"

print_section "Prophet Presentation Tests"
run_test_suite "Prophet Presentation" "-m prophet_presentation --tb=short"

print_section "Prophet Module Tests"
run_test_suite "Prophet Module" "-m prophet_module --tb=short"

# Performance Tests
print_section "Performance Tests"
echo -e "${YELLOW}⏱️  Running performance benchmarks...${NC}"
run_test_suite "Performance Tests" "-m performance --benchmark-only --benchmark-sort=mean" || echo "No performance tests found"

# Full Test Suite with Coverage
print_section "Full Test Suite with Coverage"
run_test_suite "Full Coverage" "--cov=modules --cov-report=html:$REPORTS_DIR/htmlcov --cov-report=xml:$REPORTS_DIR/coverage.xml --cov-report=term-missing --html=$REPORTS_DIR/test_report.html --self-contained-html"

# Test Quality Metrics
print_section "Test Quality Metrics"
echo "Generating test quality report..."

# Count tests by type
unit_tests=$(find tests -name "test_*.py" -exec grep -l "pytest.mark.unit" {} \; 2>/dev/null | wc -l)
integration_tests=$(find tests -name "test_*.py" -exec grep -l "pytest.mark.integration" {} \; 2>/dev/null | wc -l)
total_test_files=$(find tests -name "test_*.py" | wc -l)
total_test_functions=$(grep -r "def test_" tests/ | wc -l)

echo -e "${BLUE}📊 Test Statistics:${NC}"
echo "  Total test files: $total_test_files"
echo "  Total test functions: $total_test_functions"
echo "  Unit test files: $unit_tests"
echo "  Integration test files: $integration_tests"

# Coverage Summary
if [ -f "$REPORTS_DIR/coverage.xml" ]; then
    echo -e "\n${BLUE}📈 Coverage Summary:${NC}"
    echo "  Detailed HTML report: $REPORTS_DIR/htmlcov/index.html"
    echo "  XML report: $REPORTS_DIR/coverage.xml"
fi

# Test Report
if [ -f "$REPORTS_DIR/test_report.html" ]; then
    echo -e "\n${BLUE}📋 Test Report:${NC}"
    echo "  HTML report: $REPORTS_DIR/test_report.html"
fi

print_section "Test Categories Available"
echo "Use these commands to run specific test categories:"
echo -e "${GREEN}pytest -m unit${NC}              # Unit tests only"
echo -e "${GREEN}pytest -m integration${NC}       # Integration tests only" 
echo -e "${GREEN}pytest -m prophet_core${NC}      # Prophet core business logic"
echo -e "${GREEN}pytest -m prophet_presentation${NC} # Prophet visualization layer"
echo -e "${GREEN}pytest -m performance${NC}       # Performance benchmarks"
echo -e "${GREEN}pytest -m slow${NC}              # Long-running tests"
echo -e "${GREEN}pytest -k 'test_forecast'${NC}   # Tests containing 'forecast' in name"

print_section "Continuous Integration"
echo "For CI/CD pipelines, use:"
echo -e "${GREEN}bash tests/run_tests.sh${NC}     # Full test suite"
echo -e "${GREEN}pytest --cov=modules --cov-fail-under=80${NC} # With coverage threshold"

echo -e "\n${GREEN}🎉 Test automation script completed!${NC}"
echo -e "${BLUE}📁 Check reports in: $REPORTS_DIR${NC}"
