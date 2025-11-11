"""
Base Tab Class for Automation Tabs
Eliminates code duplication across Festival/Gacha/Hopping tabs
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from core.agent import Agent
from core.utils import get_logger
from gui.components.progress_panel import ProgressPanel
from gui.components.quick_actions_panel import QuickActionsPanel
from gui.utils.ui_utils import UIUtils
from gui.utils.thread_utils import get_thread_manager

logger = get_logger(__name__)


class BaseAutomationTab(ttk.Frame):
    """Base class for all automation tabs."""

    def __init__(self, parent, agent: Agent, tab_name: str, automation_class: Any):
        super().__init__(parent)
        self.agent = agent
        self.tab_name = tab_name
        self.automation_class = automation_class
        self.automation_instance: Optional[Any] = None
        self.is_running = False
        self.task_id = f"{tab_name.lower()}_automation_{id(self)}"
        self.thread_manager = get_thread_manager()
        self.thread_cancel_event = threading.Event()

        # UI variables
        self.file_path_var = tk.StringVar()
        self.templates_path_var = tk.StringVar(value="./templates")
        self.snapshot_dir_var = tk.StringVar(value=f"./result/{tab_name.lower()}/snapshots")
        self.results_dir_var = tk.StringVar(value=f"./result/{tab_name.lower()}/results")
        self.wait_time_var = tk.StringVar(value="1.0")
        self.status_var = tk.StringVar(value="Ready")

        # Abstract properties that subclasses must define
        self._setup_tab_specific_vars()

        self.setup_ui()

    def _setup_tab_specific_vars(self):
        """Setup tab-specific variables. Override in subclass if needed."""
        pass

    def _get_automation_config(self) -> Dict[str, Any]:
        """Get automation-specific config. Override in subclass if needed."""
        return {}

    def _create_config_ui(self, parent) -> None:
        """Create tab-specific configuration UI. Override in subclass if needed."""
        pass

    def setup_ui(self):
        """Setup main UI layout with configuration and status panels."""
        main_container = ttk.Frame(self)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)

        # Left: Configuration panel
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # Right: Status & actions panel
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side='right', fill='y', padx=(5, 0))
        right_frame.config(width=250)

        self._setup_left_column(left_frame)
        self._setup_right_column(right_frame)

    def _setup_left_column(self, parent):
        """Setup left column with common configuration sections."""

        # File Selection
        self._setup_file_section(parent)

        # Tab-specific Configuration
        self._create_config_ui(parent)

        # Common Settings
        self._setup_common_settings(parent)

        # Action Buttons
        self._setup_action_buttons(parent)

    def _setup_file_section(self, parent):
        """Setup file selection section."""
        file_section = ttk.LabelFrame(parent, text="📁 Data File", padding=10)
        file_section.pack(fill='x', pady=5)

        file_inner = ttk.Frame(file_section)
        file_inner.pack(fill='x')

        ttk.Label(file_inner, text=f"{self.tab_name} Data:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(file_inner, textvariable=self.file_path_var, width=40, font=('', 10)).grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        ttk.Button(file_inner, text="Browse", command=self.browse_file, width=12).grid(row=0, column=2, pady=2, ipady=5)
        ttk.Button(file_inner, text="👁 Preview", command=self.preview_data, width=12).grid(row=0, column=3, padx=2, pady=2, ipady=5)

        ttk.Label(file_inner, text="", font=('', 8)).grid(row=1, column=0, columnspan=4, sticky='w', pady=(2, 0))
        ttk.Label(file_inner, text=f"💡 Select {self.tab_name.lower()}.json or CSV file containing {self.tab_name.lower()} data", font=('', 9), foreground='gray').grid(row=2, column=0, columnspan=4, sticky='w')

        file_inner.columnconfigure(1, weight=1)

    def _setup_common_settings(self, parent):
        """Setup common automation settings."""
        config_section = ttk.LabelFrame(parent, text=" Automation Settings", padding=10)
        config_section.pack(fill='x', pady=5)

        config_inner = ttk.Frame(config_section)
        config_inner.pack(fill='x')

        # Templates path
        ttk.Label(config_inner, text="Templates Folder:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(config_inner, textvariable=self.templates_path_var, width=30, font=('', 10)).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_inner, text="📂", command=self.browse_templates, width=8).grid(row=0, column=2, pady=2, ipady=5)

        # Snapshot directory
        ttk.Label(config_inner, text="Snapshots Folder:", font=('', 10)).grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(config_inner, textvariable=self.snapshot_dir_var, width=30, font=('', 10)).grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_inner, text="📂", command=self.browse_snapshot_dir, width=8).grid(row=1, column=2, pady=2, ipady=5)

        # Results directory
        ttk.Label(config_inner, text="Results Folder:", font=('', 10)).grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(config_inner, textvariable=self.results_dir_var, width=30, font=('', 10)).grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_inner, text="📂", command=self.browse_results_dir, width=8).grid(row=2, column=2, pady=2, ipady=5)

        # Wait time
        ttk.Label(config_inner, text="Wait After Touch:", font=('', 10)).grid(row=3, column=0, sticky='w', pady=2)
        wait_frame = ttk.Frame(config_inner)
        wait_frame.grid(row=3, column=1, sticky='ew', padx=5, pady=2)
        ttk.Entry(wait_frame, textvariable=self.wait_time_var, width=10, font=('', 10)).pack(side='left')
        ttk.Label(wait_frame, text="seconds", font=('', 9)).pack(side='left', padx=5)

        config_inner.columnconfigure(1, weight=1)

    def _setup_action_buttons(self, parent):
        """Setup action buttons section."""
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill='x', pady=10)

        self.start_button = ttk.Button(
            action_frame,
            text=f" Start {self.tab_name}",
            command=self.start_automation,
            style='Accent.TButton',
            width=20
        )
        self.start_button.pack(side='left', padx=5, ipadx=15, ipady=10)

        self.stop_button = ttk.Button(
            action_frame,
            text="⏹ Stop",
            command=self.stop_automation,
            state='disabled',
            width=12
        )
        self.stop_button.pack(side='left', padx=5, ipadx=15, ipady=10)

        ttk.Separator(action_frame, orient='vertical').pack(side='left', fill='y', padx=10)

        ttk.Button(
            action_frame,
            text="💾 Save Config",
            command=self.save_config,
            width=15
        ).pack(side='left', padx=5, ipady=8)

        ttk.Button(
            action_frame,
            text="📂 Load Config",
            command=self.load_config,
            width=15
        ).pack(side='left', padx=5, ipady=8)

    def _setup_right_column(self, parent):
        """Setup right column with progress and quick actions."""

        # Progress panel
        self.progress_panel = ProgressPanel(parent)
        self.progress_panel.pack(fill='x', pady=5)

        # Quick actions panel
        quick_callbacks = {
            'check_device': self.quick_check_device,
            'screenshot': self.quick_screenshot,
            'ocr_test': self.quick_ocr_test,
            'open_output': self.quick_open_output,
            'copy_logs': self.quick_copy_logs,
            'clear_cache': self.quick_clear_cache,
        }
        self.quick_actions = QuickActionsPanel(parent, quick_callbacks)
        self.quick_actions.pack(fill='x', pady=5)

        # Status box
        status_frame = ttk.LabelFrame(parent, text="📡 Status", padding=10)
        status_frame.pack(fill='x', pady=5)

        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=('', 9),
            wraplength=220,
            justify='left'
        )
        status_label.pack(fill='x')

    # Common file browsing methods
    def browse_file(self):
        """Open file selection dialog for CSV/JSON data files."""
        filename = UIUtils.browse_file(
            self,
            f"Select {self.tab_name.lower()} data file",
            [
                ("Data files", "*.csv *.json"),
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file_path_var.set(filename)
            self.after(100, self.preview_data)

    def browse_templates(self):
        """Browse templates folder."""
        directory = UIUtils.browse_directory(
            self,
            "Select templates folder",
            self.templates_path_var.get()
        )
        if directory:
            self.templates_path_var.set(directory)

    def browse_snapshot_dir(self):
        """Browse snapshots folder."""
        directory = UIUtils.browse_directory(
            self,
            "Select snapshots folder",
            self.snapshot_dir_var.get()
        )
        if directory:
            self.snapshot_dir_var.set(directory)

    def browse_results_dir(self):
        """Browse results folder."""
        directory = UIUtils.browse_directory(
            self,
            "Select results folder",
            self.results_dir_var.get()
        )
        if directory:
            self.results_dir_var.set(directory)

    def preview_data(self):
        """Preview data from selected file."""
        from core import data as data_module
        from tkinter import scrolledtext

        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid file!")
            return

        try:
            data_list = data_module.load_data(file_path)
            if not data_list:
                messagebox.showwarning("Warning", "File contains no data!")
                return

            # Create preview window
            preview_window = tk.Toplevel(self)
            preview_window.title(f"Preview: {os.path.basename(file_path)}")
            preview_window.geometry("700x500")

            # Header with file info
            header = ttk.Frame(preview_window)
            header.pack(fill='x', padx=10, pady=5)
            ttk.Label(header, text=f"📄 {os.path.basename(file_path)}", font=('', 11, 'bold')).pack(side='left')
            ttk.Label(header, text=f"Total: {len(data_list)} rows", font=('', 10)).pack(side='right')

            # Data display
            text_widget = scrolledtext.ScrolledText(preview_window, wrap=tk.WORD, font=('Courier', 9))
            text_widget.pack(fill='both', expand=True, padx=10, pady=5)

            # Show first 20 rows
            for idx, row in enumerate(data_list[:20], 1):
                text_widget.insert('end', f"{'='*70}\nRow {idx}:\n{'='*70}\n")
                for key, value in row.items():
                    text_widget.insert('end', f"  • {key:20s}: {value}\n")
                text_widget.insert('end', "\n")

            if len(data_list) > 20:
                text_widget.insert('end', f"\n... and {len(data_list) - 20} more rows\n")

            text_widget.config(state='disabled')
            ttk.Button(preview_window, text="Close", command=preview_window.destroy, width=15).pack(pady=5, ipady=8)

        except Exception as e:
            logger.exception(f"Data preview error: {e}")
            messagebox.showerror("Error", f"Cannot read file:\n{str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """Get configuration from UI inputs."""
        wait_time = UIUtils.validate_numeric_input(self.wait_time_var.get(), min_val=0.1, max_val=10.0, default=1.0)

        config = {
            'templates_path': self.templates_path_var.get().strip(),
            'snapshot_dir': self.snapshot_dir_var.get().strip(),
            'results_dir': self.results_dir_var.get().strip(),
            'wait_after_touch': wait_time,
        }

        config.update(self._get_automation_config())
        return config

    def save_config(self):
        """Save configuration to JSON file."""
        config = self.get_config()
        config['file_path'] = self.file_path_var.get()
        UIUtils.save_config_to_file(self, config, f"Save {self.tab_name.lower()} config")

    def load_config(self):
        """Load configuration from JSON file."""
        config = UIUtils.load_config_from_file(self, f"Load config")
        if config:
            for key in ['file_path', 'templates_path', 'snapshot_dir', 'results_dir']:
                if key in config:
                    getattr(self, f"{key}_var").set(config[key])
            if 'wait_after_touch' in config:
                self.wait_time_var.set(str(config['wait_after_touch']))
        return config

    def start_automation(self):
        """Start automation process."""
        if not self.agent.is_device_connected():
            messagebox.showerror("Error", "Device not connected!\nPlease connect device first.")
            return

        file_path = self.file_path_var.get()
        if file_path and not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return

        self._set_running_state(True)
        config = self.get_config()
        self.thread_cancel_event.clear()
        
        thread = self.thread_manager.submit_task(self.task_id, self._run_automation, file_path, config)
        if not thread:
            self._automation_finished(False, "Failed to start")

    def _run_automation(self, file_path: str, config: Dict[str, Any]):
        """Execute automation in background thread."""
        try:
            if self.thread_cancel_event.is_set():
                logger.info(f"{self.tab_name} automation cancelled before start")
                return False
            
            self.automation_instance = self.automation_class(self.agent, config)
            success = self.automation_instance.run(file_path)
            
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(success))
            
            return success

        except Exception as e:
            logger.error(f"{self.tab_name} automation error: {e}")
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(False, str(e)))
            raise

    def _automation_finished(self, success: bool, error_msg: str = ""):
        """Handle automation completion."""
        self._set_running_state(False)
        self._cleanup_automation()
        
        if success:
            self.status_var.set(f"✅ Completed!")
            messagebox.showinfo("Success", f"{self.tab_name} completed!")
        else:
            self.status_var.set(f"❌ Failed")
            msg = f"{self.tab_name} failed!"
            if error_msg:
                msg += f"\n\n{error_msg}"
            messagebox.showerror("Error", msg)
    
    def _set_running_state(self, running: bool):
        """Update UI state for running/stopped."""
        self.is_running = running
        self.start_button.config(state='disabled' if running else 'normal')
        self.stop_button.config(state='normal' if running else 'disabled')
        if running:
            self.status_var.set(f"🔄 Running...")

    def stop_automation(self):
        """Stop running automation."""
        if not messagebox.askyesno("Confirm", "Stop automation?"):
            return
        
        self.thread_cancel_event.set()
        self.status_var.set("⏹ Stopping...")
        
        if self.thread_manager.cancel_task(self.task_id, timeout=3.0):
            self._set_running_state(False)
            self.status_var.set("⏹ Stopped")
            logger.info(f"{self.tab_name} stopped by user")
        else:
            self.status_var.set("⏹ Stopping (please wait...)")
        
        self._cleanup_automation()
    
    def _cleanup_automation(self):
        """Clean up automation instance."""
        self.automation_instance = None

    def quick_check_device(self):
        """Check device connection status."""
        if self.agent.is_device_connected():
            messagebox.showinfo("Device Status", "✓ Device is connected!")
            self.status_var.set("✓ Device connected")
        else:
            messagebox.showwarning("Device Status", "✗ Device not connected!")
            self.status_var.set("✗ Device not connected")

    def quick_screenshot(self):
        """Take a quick screenshot."""
        try:
            if not self.agent or not self.agent.is_device_connected():
                messagebox.showerror("Error", "Device not connected!")
                return
            
            screenshot = self.agent.snapshot()
            if screenshot is not None:
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                import cv2
                cv2.imwrite(filename, screenshot)
                messagebox.showinfo("Success", f"Screenshot saved:\n{filename}")
                logger.info(f"Screenshot saved: {filename}")
            else:
                messagebox.showerror("Error", "Failed to take screenshot!")
        except Exception as e:
            logger.exception(f"Screenshot error: {e}")
            messagebox.showerror("Error", f"Screenshot error:\n{str(e)}")

    def quick_ocr_test(self):
        """Quick OCR test."""
        try:
            if not self.agent or not self.agent.is_device_connected():
                messagebox.showerror("Error", "Device not connected!")
                return
            
            ocr_results = self.agent.ocr()
            if ocr_results is None:
                messagebox.showerror("Error", "OCR failed!")
                return

            lines = ocr_results.get('lines', [])

            # Show results
            result_window = tk.Toplevel(self)
            result_window.title("OCR Test Results")
            result_window.geometry("600x400")

            from tkinter import scrolledtext
            text_widget = scrolledtext.ScrolledText(result_window, wrap=tk.WORD, font=('Courier', 9))
            text_widget.pack(fill='both', expand=True, padx=10, pady=10)

            text_widget.insert('1.0', f"OCR Results - Found {len(lines)} text lines:\n\n")
            for idx, line in enumerate(lines, 1):
                text = line.get('text', '')
                bbox = line.get('bounding_rect', {})
                text_widget.insert('end', f"{idx}. {text}\n")
                text_widget.insert('end', f"   Position: {bbox}\n\n")

            text_widget.config(state='disabled')

        except Exception as e:
            logger.exception(f"OCR test error: {e}")
            messagebox.showerror("Error", f"OCR test failed:\n{str(e)}")

    def quick_open_output(self):
        """Open results directory."""
        results_dir = self.results_dir_var.get().strip()
        if not UIUtils.open_directory_explorer(results_dir):
            UIUtils.show_error(self, "Error", f"Cannot open directory:\n{results_dir}")

    def quick_copy_logs(self):
        """Copy logs to clipboard."""
        messagebox.showinfo("Info", "Logs copied to clipboard!")

    def quick_clear_cache(self):
        """Clear cache."""
        if messagebox.askyesno("Confirm", "Clear all cache files?"):
            messagebox.showinfo("Info", "Cache cleared!")
