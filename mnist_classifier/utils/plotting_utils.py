"""Enhanced plotting utilities with automatic saving and non-blocking display."""

import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_plot_with_title(fig, model_name: str, plot_type: str = "Training Progress"):
    """Add a main title to the plot showing the model being tested.
    
    Args:
        fig: Matplotlib figure object
        model_name: Name of the model being tested
        plot_type: Type of plot (e.g., "Training Progress", "Results")
    """
    fig.suptitle(f'{model_name} {plot_type}', fontsize=16, fontweight='bold')


def save_and_show_plot(fig, base_filename: str, plots_dir: str = "plots", 
                      save_always: bool = True, show_plot: bool = True):
    """Save plot with timestamp and show without blocking.
    
    Args:
        fig: Matplotlib figure object
        base_filename: Base name for the file (without extension)
        plots_dir: Directory to save plots
        save_always: Whether to always save the plot
        show_plot: Whether to display the plot
        
    Returns:
        Path to saved file
    """
    if save_always:
        # Create plots directory
        plots_path = Path(plots_dir)
        plots_path.mkdir(exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
        filename = f"{base_filename}_{timestamp}.png"
        plot_path = plots_path / filename
        
        # Save plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {plot_path}")
    
    if show_plot:
        # Show plot without blocking execution
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure plot displays
    
    plt.close()
    
    return plot_path if save_always else None


def configure_matplotlib_for_non_blocking():
    """Configure matplotlib for non-blocking display."""
    # Set backend that supports non-blocking display
    import matplotlib
    matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on system
    
    # Configure plot display settings
    plt.ion()  # Turn on interactive mode


def create_timestamped_filename(base_name: str, model_name: str = "", epoch: int = None) -> str:
    """Create a timestamped filename.
    
    Args:
        base_name: Base name for the file
        model_name: Name of the model (optional)
        epoch: Epoch number (optional)
        
    Returns:
        Formatted filename with timestamp
    """
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
    
    if model_name and epoch is not None:
        return f"{model_name}_{base_name}_epoch_{epoch}_{timestamp}"
    elif model_name:
        return f"{model_name}_{base_name}_{timestamp}"
    else:
        return f"{base_name}_{timestamp}"