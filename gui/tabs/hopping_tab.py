"""
Hopping Tab for GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from typing import Dict, Any

from automations.hopping import HoppingAutomation
from core.agent import Agent
from core.utils import get_logger
from gui.components.base_tab import BaseAutomationTab
from gui.utils.ui_utils import UIUtils

logger = get_logger(__name__)


class HoppingTab(BaseAutomationTab):
    """Tab for Hopping Automation with improved UI."""

    def __init__(self, parent, agent: Agent):
        super().__init__(parent, agent, "Hopping", HoppingAutomation)

    def _setup_tab_specific_vars(self):
        """Setup Hopping-specific variables."""
        self.num_hops_var = tk.StringVar(value="5")
        self.loading_wait_var = tk.StringVar(value="5.0")
        self.results_var = tk.StringVar(value="No results yet")

    def _create_config_ui(self, parent):
        """Create Hopping-specific configuration UI."""
        # Hop Configuration
        config_section = ttk.LabelFrame(parent, text=" Hopping Settings", padding=10)
        config_section.pack(fill='x', pady=5)

        config_inner = ttk.Frame(config_section)
        config_inner.pack(fill='x')

        # Number of hops
        ttk.Label(config_inner, text="Number of Hops:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(config_inner, textvariable=self.num_hops_var, width=10, font=('', 10)).grid(row=0, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(config_inner, text="💡 Each hop will take you to a different world",
                 font=('', 9), foreground='gray').grid(row=1, column=0, columnspan=2, sticky='w', pady=(5, 0))

        # Loading wait time
        ttk.Label(config_inner, text="Loading Wait:", font=('', 10)).grid(row=2, column=0, sticky='w', pady=2)
        loading_frame = ttk.Frame(config_inner)
        loading_frame.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        ttk.Entry(loading_frame, textvariable=self.loading_wait_var, width=10, font=('', 10)).pack(side='left')
        ttk.Label(loading_frame, text="seconds", font=('', 9)).pack(side='left', padx=5)

    def _get_automation_config(self) -> Dict[str, Any]:
        """Get Hopping-specific automation config."""
        num_hops = UIUtils.validate_integer_input(self.num_hops_var.get(), min_val=1, max_val=50, default=5)
        loading_wait = UIUtils.validate_numeric_input(self.loading_wait_var.get(), min_val=1.0, max_val=30.0, default=5.0)

        return {
            'num_hops': num_hops,
            'loading_wait': loading_wait,
        }

    def _setup_right_column(self, parent):
        """Override to add Hopping-specific results panel."""
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
        """Load Hopping configuration."""
        config = super().load_config()
        if config:
            if 'num_hops' in config:
                self.num_hops_var.set(str(config['num_hops']))
            if 'loading_wait' in config:
                self.loading_wait_var.set(str(config['loading_wait']))

    def start_automation(self):
        """Override to add Hopping-specific validation and progress initialization."""
        # Check device
        if not self.agent.is_device_connected():
            messagebox.showerror("Error", "Device not connected!\nPlease connect device first.")
            return

        # Get config and initialize progress
        config = self.get_config()
        self.progress_panel.start(config['num_hops'])
        self.results_var.set("Running...")

        # Set running state and start thread
        self._set_running_state(True)
        self.thread_cancel_event.clear()
        
        file_path = ""  # Hopping doesn't need file input
        thread = self.thread_manager.submit_task(self.task_id, self._run_automation, file_path, config)
        
        if not thread:
            self._automation_finished(False, "Failed to start thread")

    def _run_automation(self, file_path: str, config: Dict[str, Any]):
        """Override to handle Hopping-specific automation logic."""
        try:
            # Check for cancellation before starting
            if self.thread_cancel_event.is_set():
                logger.info("Hopping automation cancelled before start")
                return False
            
            # Initialize HoppingAutomation
            self.automation_instance = self.automation_class(self.agent, config)
            
            # Verify instance was created
            if self.automation_instance is None:
                logger.error("Failed to create Hopping automation instance")
                return False

            # Run hopping (Hopping doesn't need file input)
            success = self.automation_instance.run(config)
            
            # Call completion callback if not cancelled
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(success))
            
            return success

        except Exception as e:
            logger.error(f"Hopping automation error: {e}")
            if not self.thread_cancel_event.is_set():
                self.after(0, lambda: self._automation_finished(False, str(e)))
            raise  # Re-raise for thread manager tracking

    def _automation_finished(self, success: bool, error_msg: str = ""):
        """Override to add Hopping-specific result updates."""
        # Call parent implementation
        super()._automation_finished(success, error_msg)

        # Update results display
        if success:
            self.results_var.set("Check results folder for detailed statistics")
        else:
            self.results_var.set("Failed - check logs for details")
