#!/usr/bin/env python3
"""
Simple test script for performance optimization module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_performance_module():
    """Test the performance optimization module"""
    print("=== PHASE 4 PERFORMANCE OPTIMIZATION TEST ===")
    
    try:
        # Test 1: Import module
        print("\n1. Testing module import...")
        from modules.prophet_performance import PerformanceMonitor, AdvancedCache, DataFrameOptimizer
        print("✅ All classes imported successfully")
        
        # Test 2: Create instances
        print("\n2. Testing instance creation...")
        monitor = PerformanceMonitor()
        cache = AdvancedCache()  # Without Redis
        optimizer = DataFrameOptimizer()
        print("✅ All instances created successfully")
        
        # Test 3: Basic functionality
        print("\n3. Testing basic functionality...")
        stats = monitor.get_system_stats()
        print(f"✅ System stats: CPU={stats['cpu_percent']:.1f}%, Memory={stats['memory_percent']:.1f}%")
        
        # Test 4: Cache functionality
        print("\n4. Testing cache functionality...")
        cache.set("test", {"data": "value"})
        result = cache.get("test")
        print(f"✅ Cache test: {result}")
        
        print("\n=== PHASE 4 PERFORMANCE OPTIMIZATION - ALL TESTS PASSED! ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_performance_module()
    sys.exit(0 if success else 1)
