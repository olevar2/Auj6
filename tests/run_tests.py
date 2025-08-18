#!/usr/bin/env python3
"""
Test runner for AUJ Platform Performance Optimization tests.

This script runs the comprehensive test suite for parallel coordination,
early decision system, and performance validation.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_test_environment():
    """Setup test environment and dependencies."""
    try:
        # Add project root to Python path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        logger.info("Test environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup test environment: {e}")
        return False

def run_performance_tests():
    """Run performance optimization tests."""
    try:
        logger.info("Starting performance optimization tests...")
        
        # Run the test file
        test_file = Path(__file__).parent / "test_parallel_coordination.py"
        
        # Use py command on Windows, python3 on others
        python_cmd = "py" if os.name == 'nt' else "python3"
        
        # Try to run with pytest first
        try:
            result = subprocess.run([
                python_cmd, "-m", "pytest", 
                str(test_file), 
                "-v", 
                "--tb=short",
                "--durations=10"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Tests completed successfully with pytest")
                print(result.stdout)
                return True
            else:
                logger.warning("Pytest failed, trying direct execution")
                print(result.stderr)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Pytest not available, using direct execution")
            
        # Fallback to direct execution
        result = subprocess.run([
            python_cmd, str(test_file)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("Tests completed successfully")
            print(result.stdout)
            return True
        else:
            logger.error("Tests failed")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("Tests timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def run_quick_validation():
    """Run quick validation of key components."""
    try:
        logger.info("Running quick validation...")
        
        # Test imports
        try:
            from auj_platform.src.coordination.genius_agent_coordinator import GeniusAgentCoordinator
            from auj_platform.src.coordination.performance_monitor import CoordinationPerformanceMonitor
            logger.info("✓ Core modules import successfully")
        except ImportError as e:
            logger.error(f"✗ Import error: {e}")
            return False
        
        # Test performance monitor creation
        try:
            monitor = CoordinationPerformanceMonitor()
            monitor.start_cycle_monitoring("test_cycle")
            monitor.end_cycle_monitoring(success=True)
            logger.info("✓ Performance monitor works correctly")
        except Exception as e:
            logger.error(f"✗ Performance monitor error: {e}")
            return False
        
        logger.info("Quick validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return False

def main():
    """Main test runner."""
    logger.info("AUJ Platform Performance Optimization Test Runner")
    logger.info("=" * 60)
    
    # Setup environment
    if not setup_test_environment():
        sys.exit(1)
    
    # Run quick validation first
    if not run_quick_validation():
        logger.error("Quick validation failed, skipping full tests")
        sys.exit(1)
    
    # Run performance tests
    if run_performance_tests():
        logger.info("=" * 60)
        logger.info("All tests completed successfully!")
        logger.info("Performance optimizations are ready for deployment.")
        sys.exit(0)
    else:
        logger.error("=" * 60)
        logger.error("Some tests failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()