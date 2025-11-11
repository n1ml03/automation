"""
Festival Tab for GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from typing import Dict, Any

from automations.festivals import FestivalAutomation
from core.agent import Agent
from core.utils import get_logger
from gui.components.base_tab import BaseAutomationTab

logger = get_logger(__name__)


class FestivalTab(BaseAutomationTab):
    """Tab for Festival Automation with improved UI."""

    def __init__(self, parent, agent: Agent):
        super().__init__(parent, agent, "Festival", FestivalAutomation)

    def _setup_tab_specific_vars(self):
        """Setup Festival-specific variables."""
        self.output_file_var = tk.StringVar(value="")

    def _create_config_ui(self, parent):
        """Create Festival-specific configuration UI."""
        # Output Configuration
        output_section = ttk.LabelFrame(parent, text="💾 Output", padding=10)
        output_section.pack(fill='x', pady=5)

        output_inner = ttk.Frame(output_section)
        output_inner.pack(fill='x')

        # Output file (optional)
        ttk.Label(output_inner, text="Output File:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(output_inner, textvariable=self.output_file_var, width=30, font=('', 10)).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(output_inner, text="📂", command=self.browse_output, width=8).grid(row=0, column=2, pady=2, ipady=5)
        ttk.Label(output_inner, text="(Optional - auto-generated if empty)", font=('', 8), foreground='gray').grid(row=1, column=0, columnspan=3, sticky='w')

        output_inner.columnconfigure(1, weight=1)

    def _get_automation_config(self) -> Dict[str, Any]:
        """Get Festival-specific automation config."""
        return {}

    def browse_output(self):
        """Browse output file (optional)."""
        filename = filedialog.asksaveasfilename(
            title="Save results to (optional)",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.output_file_var.set(filename)
            logger.info(f"Output file: {filename}")

    def start_automation(self):
        """Override to add Festival-specific validation and progress initialization."""
        # Validate
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid CSV/JSON file!")
            return

        # Check device
        if not self.agent.is_device_connected():
            messagebox.showerror("Error", "Device not connected!\nPlease connect device first.")
            return

        # Load data to get count for progress tracking
        from core.data import load_data
        try:
            data_list = load_data(file_path)
            if not data_list:
                messagebox.showerror("Error", "No data in file!")
                return
            total_count = len(data_list)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load data:\n{str(e)}")
            return

        # Initialize progress
        self.progress_panel.start(total_count)

        # Call parent implementation
        super().start_automation()

    def _run_automation(self, file_path: str, config: Dict[str, Any]):
        """Override to handle Festival-specific automation logic."""
        try:
            # Check for cancellation before starting
            if self.thread_cancel_event.is_set():
                logger.info("Festival automation cancelled before start")
                return False
            
            # Initialize FestivalAutomation
            self.automation_instance = self.automation_class(self.agent, config)
            
            # Verify instance was created
            if self.automation_instance is None:
                logger.error("Failed to create Festival automation instance")
                return False

            # Get output path if specified
            output_file = self.output_file_var.get().strip()
            output_path = output_file if output_file else None

            # Run with optional output path
            if output_path:
                success = self.automation_instance.run_all_stages(file_path, output_path)
            else:
                success = self.automation_instance.run(file_path)
            
            # Call completion callback if not cancelled
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(success))
            
            return success

        except Exception as e:
            logger.error(f"Festival automation error: {e}")
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(False, str(e)))
            raise  # Re-raise for thread manager tracking
