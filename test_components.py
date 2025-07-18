#!/usr/bin/env python3
"""
Test script to verify that all components are working properly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported successfully"""
    print("Testing imports...")
    
    try:
        from eon_models import EONNode, EONLink, ModulationFormat, WSSSpec
        print("✓ eon_models imported successfully")
    except Exception as e:
        print(f"✗ eon_models import failed: {e}")
        return False
    
    try:
        from ml_qot import MLQoTEstimator, QoTFeatures
        print("✓ ml_qot imported successfully")
    except Exception as e:
        print(f"✗ ml_qot import failed: {e}")
        return False
    
    try:
        from eon_control import EONController
        print("✓ eon_control imported successfully")
    except Exception as e:
        print(f"✗ eon_control import failed: {e}")
        return False
    
    try:
        from spectrum_manager import SpectrumManager
        print("✓ spectrum_manager imported successfully")
    except Exception as e:
        print(f"✗ spectrum_manager import failed: {e}")
        return False
    
    try:
        from generate_synthetic_data import generate_network_topology
        print("✓ generate_synthetic_data imported successfully")
    except Exception as e:
        print(f"✗ generate_synthetic_data import failed: {e}")
        return False
    
    try:
        from process_lightpath_data import process_lightpath_dataset
        print("✓ process_lightpath_data imported successfully")
    except Exception as e:
        print(f"✗ process_lightpath_data import failed: {e}")
        return False
    
    try:
        from lightpath_reader import LightpathReader
        print("✓ lightpath_reader imported successfully")
    except Exception as e:
        print(f"✗ lightpath_reader import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of components"""
    print("\nTesting basic functionality...")
    
    try:
        from eon_models import EONNode, EONLink, ModulationFormat, WSSSpec
        
        # Test EONNode creation
        wss = WSSSpec()
        node = EONNode("test_node", wss)
        print("✓ EONNode created successfully")
        
        # Test EONLink creation
        link = EONLink(length=100.0)
        print("✓ EONLink created successfully")
        
        # Test ModulationFormat
        mod = ModulationFormat.QPSK
        print("✓ ModulationFormat works")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    try:
        from ml_qot import MLQoTEstimator
        
        # Test MLQoTEstimator creation
        estimator = MLQoTEstimator(model_dir="test_models")
        print("✓ MLQoTEstimator created successfully")
        
    except Exception as e:
        print(f"✗ MLQoTEstimator test failed: {e}")
        return False
    
    try:
        import networkx as nx
        from eon_control import EONController
        
        # Create a simple test graph
        G = nx.Graph()
        G.add_edge("A", "B", length=100.0)
        G.add_edge("B", "C", length=150.0)
        
        # Test EONController creation
        controller = EONController(G)
        print("✓ EONController created successfully")
        
    except Exception as e:
        print(f"✗ EONController test failed: {e}")
        return False
    
    try:
        import networkx as nx
        from spectrum_manager import SpectrumManager
        
        # Create a simple test graph
        G = nx.Graph()
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        
        # Test SpectrumManager creation
        sm = SpectrumManager(G)
        print("✓ SpectrumManager created successfully")
        
    except Exception as e:
        print(f"✗ SpectrumManager test failed: {e}")
        return False
    
    return True

def test_synthetic_data_generation():
    """Test synthetic data generation"""
    print("\nTesting synthetic data generation...")
    
    try:
        from generate_synthetic_data import generate_network_topology, generate_training_samples
        
        # Generate a small test network
        G = generate_network_topology(num_nodes=5, avg_degree=2.0)
        print(f"✓ Generated network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Generate a few training samples
        samples = generate_training_samples(G, num_samples=10)
        print(f"✓ Generated {len(samples)} training samples")
        
    except Exception as e:
        print(f"✗ Synthetic data generation test failed: {e}")
        return False
    
    return True

def cleanup():
    """Clean up test files"""
    import shutil
    
    # Remove test models directory if it exists
    test_models_dir = Path("test_models")
    if test_models_dir.exists():
        shutil.rmtree(test_models_dir)
        print("✓ Cleaned up test files")

def main():
    """Run all tests"""
    print("ML-QoT Component Test Suite")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test synthetic data generation
    if not test_synthetic_data_generation():
        success = False
    
    # Cleanup
    cleanup()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! Components are working properly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
