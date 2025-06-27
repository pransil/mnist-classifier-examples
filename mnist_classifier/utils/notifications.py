"""Notification utilities for user interaction."""

import os
import sys


def ring_bell(count=1):
    """Ring system bell for audio notification.
    
    Args:
        count: Number of bells to ring
    """
    bell_string = '\a' * count
    os.system(f"echo -e '{bell_string}'")


def prompt_with_bell(message):
    """Prompt user for input with bell notification.
    
    Args:
        message: The prompt message to display
        
    Returns:
        User's input response
    """
    ring_bell(1)
    return input(message)


def data_fallback_warning(reason, fallback_type="synthetic data"):
    """Show clear warning when real data can't be used.
    
    Args:
        reason: Reason why real data isn't available
        fallback_type: Type of fallback being offered
        
    Returns:
        True if user approves fallback, False otherwise
    """
    print(f"⚠️  REAL DATA NOT AVAILABLE: {reason}")
    print("🔔🔔🔔 DATA FALLBACK WARNING!")
    ring_bell(3)
    
    response = prompt_with_bell(f"Real data unavailable. Use {fallback_type} instead? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ Stopping execution as requested.")
        return False
    
    return True


def confirm_action(message):
    """Ask for user confirmation with bell.
    
    Args:
        message: Confirmation message
        
    Returns:
        True if user confirms, False otherwise
    """
    response = prompt_with_bell(f"{message} (y/n): ")
    return response.lower() == 'y'