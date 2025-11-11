"""
Common UI Utilities for GUI Components
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from typing import Dict, Any, Optional, Callable
from core.utils import get_logger

logger = get_logger(__name__)


class UIUtils:
    """Utility class for common UI operations."""

    @staticmethod
    def browse_file(parent: tk.Widget, title: str, filetypes: list, initial_dir: str = "") -> Optional[str]:
        """Common file browsing dialog with error handling."""
        try:
            filename = filedialog.askopenfilename(
                parent=parent,
                title=title,
                filetypes=filetypes,
                initialdir=initial_dir or None
            )
            if filename:
                logger.info(f"Selected file: {filename}")
                return filename
        except Exception as e:
            logger.error(f"File dialog error: {e}")
            messagebox.showerror("Error", f"Cannot open file dialog:\n{str(e)}")
        return None

    @staticmethod
    def browse_directory(parent: tk.Widget, title: str, initial_dir: str = "") -> Optional[str]:
        """Common directory browsing dialog with error handling."""
        try:
            directory = filedialog.askdirectory(
                parent=parent,
                title=title,
                initialdir=initial_dir or None
            )
            if directory:
                logger.info(f"Selected directory: {directory}")
                return directory
        except Exception as e:
            logger.error(f"Directory dialog error: {e}")
            messagebox.showerror("Error", f"Cannot open directory dialog:\n{str(e)}")
        return None

    @staticmethod
    def save_file(parent: tk.Widget, title: str, defaultextension: str = "",
                  filetypes: Optional[list] = None) -> Optional[str]:
        """Common save file dialog with error handling."""
        if filetypes is None:
            filetypes = [("All files", "*.*")]

        try:
            filename = filedialog.asksaveasfilename(
                parent=parent,
                title=title,
                defaultextension=defaultextension,
                filetypes=filetypes
            )
            if filename:
                logger.info(f"Save file selected: {filename}")
                return filename
        except Exception as e:
            logger.error(f"Save file dialog error: {e}")
            messagebox.showerror("Error", f"Cannot open save dialog:\n{str(e)}")
        return None

    @staticmethod
    def validate_numeric_input(value: str, min_val: Optional[float] = None, max_val: Optional[float] = None,
                             default: float = 0.0) -> float:
        """Validate numeric input with bounds checking."""
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                logger.warning(f"Value {num} below minimum {min_val}, using default {default}")
                return default
            if max_val is not None and num > max_val:
                logger.warning(f"Value {num} above maximum {max_val}, using default {default}")
                return default
            return num
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric value '{value}', using default {default}")
            return default

    @staticmethod
    def validate_integer_input(value: str, min_val: Optional[int] = None, max_val: Optional[int] = None,
                             default: int = 0) -> int:
        """Validate integer input with bounds checking."""
        try:
            num = int(float(value))  # Handle float strings like "1.0"
            if min_val is not None and num < min_val:
                logger.warning(f"Value {num} below minimum {min_val}, using default {default}")
                return default
            if max_val is not None and num > max_val:
                logger.warning(f"Value {num} above maximum {max_val}, using default {default}")
                return default
            return num
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value '{value}', using default {default}")
            return default

    @staticmethod
    def save_config_to_file(parent: tk.Widget, config: Dict[str, Any], title: str = "Save config",
                           defaultextension: str = ".json") -> bool:
        """Save configuration to JSON file with error handling."""
        filename = UIUtils.save_file(parent, title, defaultextension,
                                   [("JSON files", "*.json"), ("All files", "*.*")])
        if not filename:
            return False

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", f"Config saved to:\n{filename}")
            logger.info(f"Config saved: {filename}")
            return True
        except Exception as e:
            logger.error(f"Cannot save config: {e}")
            messagebox.showerror("Error", f"Cannot save config:\n{str(e)}")
            return False

    @staticmethod
    def load_config_from_file(parent: tk.Widget, title: str = "Load config") -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file with error handling."""
        filename = UIUtils.browse_file(parent, title,
                                     [("JSON files", "*.json"), ("All files", "*.*")])
        if not filename:
            return None

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)
            messagebox.showinfo("Success", "Config loaded successfully!")
            logger.info(f"Config loaded: {filename}")
            return config
        except Exception as e:
            logger.error(f"Cannot load config: {e}")
            messagebox.showerror("Error", f"Cannot load config:\n{str(e)}")
            return None

    @staticmethod
    def ensure_directory_exists(directory: str) -> bool:
        """Ensure directory exists, create if necessary."""
        if not directory:
            return False

        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            return True
        except Exception as e:
            logger.error(f"Cannot create directory {directory}: {e}")
            return False

    @staticmethod
    def open_directory_explorer(directory: str) -> bool:
        """Open directory in Windows Explorer."""
        if not UIUtils.ensure_directory_exists(directory):
            return False

        try:
            import subprocess
            subprocess.call(['explorer', os.path.abspath(directory)])
            logger.info(f"Opened directory: {directory}")
            return True
        except Exception as e:
            logger.error(f"Cannot open directory {directory}: {e}")
            return False

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def calculate_eta(processed: int, total: int, elapsed_time: float) -> str:
        """Calculate estimated time of arrival."""
        if processed <= 0 or total <= 0:
            return "--:--:--"

        try:
            avg_time = elapsed_time / processed
            remaining = (total - processed) * avg_time
            return UIUtils.format_time(remaining)
        except (ZeroDivisionError, OverflowError):
            return "--:--:--"

    @staticmethod
    def safe_callback(callback: Callable, *args, **kwargs):
        """Safely execute callback with error handling."""
        try:
            return callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Callback error: {e}")
            return None

    @staticmethod
    def confirm_action(parent: tk.Widget, title: str, message: str) -> bool:
        """Show confirmation dialog."""
        return messagebox.askyesno(title, message, parent=parent)

    @staticmethod
    def show_info(parent: tk.Widget, title: str, message: str):
        """Show info message dialog."""
        messagebox.showinfo(title, message, parent=parent)

    @staticmethod
    def show_warning(parent: tk.Widget, title: str, message: str):
        """Show warning message dialog."""
        messagebox.showwarning(title, message, parent=parent)

    @staticmethod
    def show_error(parent: tk.Widget, title: str, message: str):
        """Show error message dialog."""
        messagebox.showerror(title, message, parent=parent)
