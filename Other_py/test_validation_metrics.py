"""
Quick Test Script for New Validation Metrics
Run this to verify the accuracy tracking enhancement works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that new functions can be imported."""
    print("="*60)
    print("Testing New Validation Metrics Import")
    print("="*60)
    
    try:
        from HAN import compute_accuracy, plot_training_metrics_enhanced
        print("✅ compute_accuracy imported successfully!")
        print("✅ plot_training_metrics_enhanced imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_compute_accuracy():
    """Test compute_accuracy function with dummy data."""
    print("\n" + "="*60)
    print("Testing compute_accuracy Function")
    print("="*60)
    
    try:
        import torch
        import numpy as np
        from HAN import HANPP, compute_accuracy
        
        # Create dummy model
        model = HANPP(
            in_dim=10,
            hidden_dim=16,
            out_dim=16,
            metapath_names=["P-O-P"],
            num_heads=2,
            num_organs=5,
            num_severity=4,
            dropout=0.1
        )
        
        # Create dummy data
        patient_feats = torch.randn(20, 10)
        labels_severity = torch.randint(0, 4, (20, 5))
        neighbor_dict = {"P-O-P": {i: [i] for i in range(20)}}
        idx_set = set(range(10))
        
        # Compute accuracy
        acc_dict = compute_accuracy(
            model, patient_feats, labels_severity, 
            neighbor_dict, idx_set
        )
        
        print(f"✅ Accuracy computation successful!")
        print(f"   Overall accuracy: {acc_dict['overall_accuracy']:.4f}")
        print(f"   Mean organ accuracy: {acc_dict['mean_organ_accuracy']:.4f}")
        print(f"   Number of organs: {len(acc_dict['per_organ_accuracy'])}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_plotting():
    """Test enhanced plotting function with dummy data."""
    print("\n" + "="*60)
    print("Testing Enhanced Plotting Function")
    print("="*60)
    
    try:
        from HAN import plot_training_metrics_enhanced
        import numpy as np
        import os
        
        # Create dummy training data
        epochs = 20
        train_losses = list(np.random.uniform(0.5, 1.0, epochs))
        val_losses = list(np.random.uniform(0.6, 1.1, epochs//2))
        val_f1 = list(np.random.uniform(0.5, 0.8, epochs//2))
        val_micro_f1 = list(np.random.uniform(0.5, 0.8, epochs//2))
        val_macro_f1 = list(np.random.uniform(0.5, 0.8, epochs//2))
        train_acc = list(np.random.uniform(0.6, 0.9, epochs))
        val_acc = list(np.random.uniform(0.6, 0.9, epochs//2))
        
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "test_plot.png")
        
        # Generate plot
        plot_training_metrics_enhanced(
            train_losses, val_losses, val_f1, val_micro_f1, val_macro_f1,
            train_acc, val_acc,
            "Test Model", "P-O-P", save_path
        )
        
        if os.path.exists(save_path):
            print(f"✅ Plot generated successfully!")
            print(f"   Saved to: {save_path}")
            print(f"   File size: {os.path.getsize(save_path)/1024:.1f} KB")
            return True
        else:
            print(f"❌ Plot file not created")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("HAN VALIDATION METRICS TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Import Test", test_imports()))
    
    # Test 2: Accuracy computation
    results.append(("Accuracy Computation", test_compute_accuracy()))
    
    # Test 3: Enhanced plotting
    results.append(("Enhanced Plotting", test_enhanced_plotting()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nYou're ready to use the enhanced validation metrics!")
        print("\nNext steps:")
        print("1. Read VALIDATION_GUIDE.md to understand validation process")
        print("2. Read NOTEBOOK_UPDATE_GUIDE.md for notebook integration")
        print("3. Update your train.ipynb with accuracy tracking")
        print("4. Run training and see enhanced 6-subplot visualizations!")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the error messages above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
