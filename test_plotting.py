#!/usr/bin/env python
"""Test the enhanced plotting functionality."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mnist_classifier.utils.plotting_utils import setup_plot_with_title, save_and_show_plot, create_timestamped_filename

def test_plotting():
    """Test the plotting enhancements."""
    print("🧪 Testing enhanced plotting functionality...")
    
    # Create a simple test plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Add main title
    setup_plot_with_title(fig, "MLP_Small", "Test Plot")
    
    # Create some sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax1.plot(x, y1, 'b-', label='Training Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(x, y2, 'r-', label='Validation Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for main title
    
    # Test timestamped filename
    filename = create_timestamped_filename("test_plot", "MLP_Small", epoch=5)
    print(f"📝 Generated filename: {filename}")
    
    # Save and show (with show_plot=False for testing)
    saved_path = save_and_show_plot(fig, "test_plot", show_plot=False)
    print(f"✅ Plot saved to: {saved_path}")
    
    print("✅ Plotting test complete!")

if __name__ == "__main__":
    test_plotting()