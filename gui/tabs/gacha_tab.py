"""
Gacha Tab for GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from typing import Dict, Any

from automations.gachas import GachaAutomation
from core.agent import Agent
from core.utils import get_logger
from gui.components.base_tab import BaseAutomationTab
from gui.utils.ui_utils import UIUtils

logger = get_logger(__name__)


class GachaTab(BaseAutomationTab):
    """Tab for Gacha Automation with improved UI."""

    def __init__(self, parent, agent: Agent):
        super().__init__(parent, agent, "Gacha", GachaAutomation)

    def _setup_tab_specific_vars(self):
        """Setup Gacha-specific variables."""
        self.num_pulls_var = tk.StringVar(value="10")
        self.pull_type_var = tk.StringVar(value="single")
        self.results_var = tk.StringVar(value="No results yet")

    def _create_config_ui(self, parent):
        """Create Gacha-specific configuration UI."""
        # Pull Configuration
        config_section = ttk.LabelFrame(parent, text=" Gacha Settings", padding=10)
        config_section.pack(fill='x', pady=5)

        config_inner = ttk.Frame(config_section)
        config_inner.pack(fill='x')

        # Number of pulls
        ttk.Label(config_inner, text="Number of Pulls:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(config_inner, textvariable=self.num_pulls_var, width=10, font=('', 10)).grid(row=0, column=1, sticky='w', padx=5, pady=2)

        # Pull type
        ttk.Label(config_inner, text="Pull Type:", font=('', 10)).grid(row=1, column=0, sticky='w', pady=2)
        pull_type_combo = ttk.Combobox(config_inner, textvariable=self.pull_type_var,
                                      values=["single", "multi"], state='readonly', width=8)
        pull_type_combo.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(config_inner, text="💡 Single pull = 1 ticket, Multi pull = 10 tickets",
                 font=('', 9), foreground='gray').grid(row=2, column=0, columnspan=2, sticky='w', pady=(5, 0))

    def _get_automation_config(self) -> Dict[str, Any]:
        """Get Gacha-specific automation config."""
        num_pulls = UIUtils.validate_integer_input(self.num_pulls_var.get(), min_val=1, max_val=100, default=10)

        return {
            'num_pulls': num_pulls,
            'pull_type': self.pull_type_var.get(),
        }

    def _setup_right_column(self, parent):
        """Override to add Gacha-specific results panel."""
        # Call parent implementation first
        super()._setup_right_column(parent)

        # Add results summary panel
        results_frame = ttk.LabelFrame(parent, text="📊 Results", padding=10)
        results_frame.pack(fill='x', pady=5)

        results_label = ttk.Label(
            results_frame,
            textvariable=self.results_var,
            font=('', 9),
            wraplength=220,
            justify='left'
        )
        results_label.pack(fill='x')

    # Use parent's save_config - no need to override

    def load_config(self):
        """Load Gacha configuration."""
        config = super().load_config()
        if config:
            if 'num_pulls' in config:
                self.num_pulls_var.set(str(config['num_pulls']))
            if 'pull_type' in config:
                self.pull_type_var.set(config['pull_type'])

    def start_automation(self):
        """Override to add Gacha-specific UI updates."""
        # Check device
        if not self.agent.is_device_connected():
            messagebox.showerror("Error", "Device not connected!\nPlease connect device first.")
            return

        # Get config and initialize progress
        config = self.get_config()
        self.progress_panel.start(config['num_pulls'])

        self.results_var.set("Running...")
        self.is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_var.set(f"🔄 Running {self.tab_name.lower()} automation...")

        file_path = ""  # Gacha doesn't need file input
        self.thread_cancel_event.clear()
        thread = self.thread_manager.submit_task(self.task_id, self._run_automation, file_path, config)
        
        if thread:
            logger.info(f"{self.tab_name} automation started")
        else:
            self._automation_finished(False, "Failed to start thread")

    def _run_automation(self, file_path: str, config: Dict[str, Any]):
        """Override to handle Gacha-specific automation logic."""
        try:
            # Check for cancellation before starting
            if self.thread_cancel_event.is_set():
                logger.info("Gacha automation cancelled before start")
                return False
            
            # Initialize GachaAutomation
            self.automation_instance = self.automation_class(self.agent, config)

            # Run gacha pulls (Gacha doesn't need file input)
            success = self.automation_instance.run(config)
            
            # Call completion callback if not cancelled
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(success))
            
            return success

        except Exception as e:
            logger.error(f"Gacha automation error: {e}")
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(False, str(e)))
            raise  # Re-raise for thread manager tracking

    def _automation_finished(self, success: bool, error_msg: str = ""):
        """Override to add Gacha-specific result updates."""
        # Call parent implementation
        super()._automation_finished(success, error_msg)

        # Update results display
        if success:
            self.results_var.set("Check results folder for detailed statistics")
        else:
            self.results_var.set("Failed - check logs for details")
