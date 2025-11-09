"""
Auto C-Peach - Main Entry Point
Game Automation Tool

Usage:
    python main.py              - Launch GUI
"""

import sys
import argparse
from gui.main import main as gui_main


def main():
    """Main entry point for Auto C-Peach application."""
    parser = argparse.ArgumentParser(
        description='Auto C-Peach - Game Automation Tool'
    )
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run in CLI mode (default: GUI mode)'
    )
    
    args = parser.parse_args()
    
    if args.cli:
        print("CLI mode is not yet implemented.")
        print("Please use GUI mode for now: python main.py")
        sys.exit(1)
    else:
        # Launch GUI
        gui_main()


if __name__ == '__main__':
    main()

