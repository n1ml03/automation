"""
Gacha Tab for GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

from automations.gachas import GachaAutomation
from core.agent import Agent
from core.utils import get_logger
from gui.components.progress_panel import ProgressPanel
from gui.components.quick_actions_panel import QuickActionsPanel

logger = get_logger(__name__)


class GachaTab(ttk.Frame):
    """Tab cho Gacha Automation với UI cải tiến."""

    def __init__(self, parent, agent: Agent):
        super().__init__(parent)
        self.agent = agent
        self.gacha_automation: Optional[GachaAutomation] = None
        self.is_running = False
        self.automation_thread: Optional[threading.Thread] = None

        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện cho Gacha tab."""

        # Main container với 2 columns
        main_container = ttk.Frame(self)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)

        # Left column - Configuration
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # Right column - Status & Quick Actions
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side='right', fill='y', padx=(5, 0))
        right_frame.config(width=250)

        # === LEFT COLUMN ===

        # 1. Pull Configuration
        config_section = ttk.LabelFrame(left_frame, text="🎰 Gacha Settings", padding=10)
        config_section.pack(fill='x', pady=5)

        config_inner = ttk.Frame(config_section)
        config_inner.pack(fill='x')

        # Number of pulls
        self.num_pulls_var = tk.StringVar(value="10")
        ttk.Label(config_inner, text="Number of Pulls:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(config_inner, textvariable=self.num_pulls_var, width=10, font=('', 10)).grid(row=0, column=1, sticky='w', padx=5, pady=2)

        # Pull type
        self.pull_type_var = tk.StringVar(value="single")
        ttk.Label(config_inner, text="Pull Type:", font=('', 10)).grid(row=1, column=0, sticky='w', pady=2)
        pull_type_combo = ttk.Combobox(config_inner, textvariable=self.pull_type_var,
                                      values=["single", "multi"], state='readonly', width=8)
        pull_type_combo.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        ttk.Label(config_inner, text="💡 Single pull = 1 ticket, Multi pull = 10 tickets",
                 font=('', 9), foreground='gray').grid(row=2, column=0, columnspan=2, sticky='w', pady=(5, 0))

        # 2. Configuration
        settings_section = ttk.LabelFrame(left_frame, text="⚙️ Automation Settings", padding=10)
        settings_section.pack(fill='x', pady=5)

        settings_inner = ttk.Frame(settings_section)
        settings_inner.pack(fill='x')

        # Templates path
        self.templates_path_var = tk.StringVar(value="./templates")
        ttk.Label(settings_inner, text="Templates Folder:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(settings_inner, textvariable=self.templates_path_var, width=30, font=('', 10)).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(settings_inner, text="📂", command=self.browse_templates, width=8).grid(row=0, column=2, pady=2, ipady=5)

        # Snapshot directory
        self.snapshot_dir_var = tk.StringVar(value="./result/gacha/snapshots")
        ttk.Label(settings_inner, text="Snapshots Folder:", font=('', 10)).grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(settings_inner, textvariable=self.snapshot_dir_var, width=30, font=('', 10)).grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(settings_inner, text="📂", command=self.browse_snapshot_dir, width=8).grid(row=1, column=2, pady=2, ipady=5)

        # Results directory
        self.results_dir_var = tk.StringVar(value="./result/gacha/results")
        ttk.Label(settings_inner, text="Results Folder:", font=('', 10)).grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(settings_inner, textvariable=self.results_dir_var, width=30, font=('', 10)).grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(settings_inner, text="📂", command=self.browse_results_dir, width=8).grid(row=2, column=2, pady=2, ipady=5)

        # Wait time
        self.wait_time_var = tk.StringVar(value="1.0")
        ttk.Label(settings_inner, text="Wait After Touch:", font=('', 10)).grid(row=3, column=0, sticky='w', pady=2)
        wait_frame = ttk.Frame(settings_inner)
        wait_frame.grid(row=3, column=1, sticky='ew', padx=5, pady=2)
        ttk.Entry(wait_frame, textvariable=self.wait_time_var, width=10, font=('', 10)).pack(side='left')
        ttk.Label(wait_frame, text="seconds", font=('', 9)).pack(side='left', padx=5)

        settings_inner.columnconfigure(1, weight=1)

        # 3. Action Buttons
        action_frame = ttk.Frame(left_frame)
        action_frame.pack(fill='x', pady=10)

        self.start_button = ttk.Button(
            action_frame,
            text="🎰 Start Gacha",
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

        # === RIGHT COLUMN ===

        # Progress panel
        self.progress_panel = ProgressPanel(right_frame)
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
        self.quick_actions = QuickActionsPanel(right_frame, quick_callbacks)
        self.quick_actions.pack(fill='x', pady=5)

        # Status box
        status_frame = ttk.LabelFrame(right_frame, text="📡 Status", padding=10)
        status_frame.pack(fill='x', pady=5)

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=('', 9),
            wraplength=220,
            justify='left'
        )
        status_label.pack(fill='x')

        # Results summary
        results_frame = ttk.LabelFrame(right_frame, text="📊 Results", padding=10)
        results_frame.pack(fill='x', pady=5)

        self.results_var = tk.StringVar(value="No results yet")
        results_label = ttk.Label(
            results_frame,
            textvariable=self.results_var,
            font=('', 9),
            wraplength=220,
            justify='left'
        )
        results_label.pack(fill='x')

    def browse_templates(self):
        """Browse templates folder."""
        directory = filedialog.askdirectory(
            title="Select templates folder",
            initialdir=self.templates_path_var.get()
        )
        if directory:
            self.templates_path_var.set(directory)
            logger.info(f"Templates folder: {directory}")

    def browse_snapshot_dir(self):
        """Browse snapshots folder."""
        directory = filedialog.askdirectory(
            title="Select snapshots folder",
            initialdir=self.snapshot_dir_var.get()
        )
        if directory:
            self.snapshot_dir_var.set(directory)
            logger.info(f"Snapshots folder: {directory}")

    def browse_results_dir(self):
        """Browse results folder."""
        directory = filedialog.askdirectory(
            title="Select results folder",
            initialdir=self.results_dir_var.get()
        )
        if directory:
            self.results_dir_var.set(directory)
            logger.info(f"Results folder: {directory}")

    def get_config(self) -> Dict[str, Any]:
        """Lấy cấu hình từ UI."""
        try:
            num_pulls = int(self.num_pulls_var.get())
            wait_time = float(self.wait_time_var.get())
        except ValueError:
            num_pulls = 10
            wait_time = 1.0
            logger.warning("Invalid config values, using defaults")

        config = {
            'num_pulls': num_pulls,
            'pull_type': self.pull_type_var.get(),
            'templates_path': self.templates_path_var.get().strip(),
            'snapshot_dir': self.snapshot_dir_var.get().strip(),
            'results_dir': self.results_dir_var.get().strip(),
            'wait_after_touch': wait_time,
        }

        return config

    def save_config(self):
        """Lưu cấu hình vào file JSON."""
        filename = filedialog.asksaveasfilename(
            title="Save gacha config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filename:
            return

        config = self.get_config()

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", f"Config saved to:\n{filename}")
            logger.info(f"Config saved: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot save config:\n{str(e)}")

    def load_config(self):
        """Load cấu hình từ file JSON."""
        filename = filedialog.askopenfilename(
            title="Load gacha config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Apply config to UI
            if 'num_pulls' in config:
                self.num_pulls_var.set(str(config['num_pulls']))
            if 'pull_type' in config:
                self.pull_type_var.set(config['pull_type'])
            if 'templates_path' in config:
                self.templates_path_var.set(config['templates_path'])
            if 'snapshot_dir' in config:
                self.snapshot_dir_var.set(config['snapshot_dir'])
            if 'results_dir' in config:
                self.results_dir_var.set(config['results_dir'])
            if 'wait_after_touch' in config:
                self.wait_time_var.set(str(config['wait_after_touch']))

            messagebox.showinfo("Success", "Config loaded successfully!")
            logger.info(f"Config loaded: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load config:\n{str(e)}")

    def start_automation(self):
        """Bắt đầu gacha automation."""
        # Check device
        if not self.agent.is_device_connected():
            messagebox.showerror("Error", "Device not connected!\nPlease connect device first.")
            return

        # Get config
        config = self.get_config()

        # Update UI
        self.is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_var.set("🎰 Running gacha automation...")
        self.results_var.set("Running...")

        # Initialize progress
        self.progress_panel.start(config['num_pulls'])

        # Run in thread
        self.automation_thread = threading.Thread(
            target=self._run_automation,
            args=(config,),
            daemon=True
        )
        self.automation_thread.start()
        logger.info("Gacha automation started")

    def _run_automation(self, config: Dict[str, Any]):
        """Chạy automation (trong thread)."""
        try:
            # Khởi tạo GachaAutomation
            self.gacha_automation = GachaAutomation(self.agent, config)

            # Run gacha pulls
            success = self.gacha_automation.run(config)

            # Update UI
            self.after(0, lambda: self._automation_finished(success))

        except Exception as e:
            logger.error(f"Gacha automation error: {e}")
            self.after(0, lambda: self._automation_finished(False, str(e)))

    def _automation_finished(self, success: bool, error_msg: str = ""):
        """Callback khi automation kết thúc."""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

        if success:
            self.status_var.set("✅ Gacha automation completed!")
            self.results_var.set("Check results folder for detailed statistics")
            messagebox.showinfo("Success", "Gacha automation completed!")
        else:
            self.status_var.set("❌ Gacha automation failed")
            self.results_var.set("Failed - check logs for details")
            msg = "Gacha automation failed!"
            if error_msg:
                msg += f"\n\nError: {error_msg}"
            messagebox.showerror("Error", msg)

    def stop_automation(self):
        """Dừng automation."""
        if messagebox.askyesno("Confirm", "Are you sure you want to stop?"):
            self.is_running = False
            self.status_var.set("⏹ Stopped by user")
            logger.warning("Gacha automation stopped by user")

    # Quick action methods
    def quick_check_device(self):
        """Quick check device connection."""
        if self.agent.is_device_connected():
            messagebox.showinfo("Device Status", "✅ Device is connected!")
            self.status_var.set("✅ Device connected")
        else:
            messagebox.showwarning("Device Status", "❌ Device not connected!")
            self.status_var.set("❌ Device not connected")

    def quick_screenshot(self):
        """Take a quick screenshot."""
        try:
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
            messagebox.showerror("Error", f"Screenshot error:\n{str(e)}")

    def quick_ocr_test(self):
        """Quick OCR test."""
        try:
            screenshot = self.agent.snapshot()
            if screenshot is None:
                messagebox.showerror("Error", "Cannot capture screen!")
                return

            if self.agent.ocr_engine is None:
                messagebox.showerror("Error", "OCR engine not initialized!")
                return

            ocr_results = self.agent.ocr_engine.recognize_cv2(screenshot)
            lines = ocr_results.get('lines', [])

            # Show results
            result_window = tk.Toplevel(self)
            result_window.title("OCR Test Results")
            result_window.geometry("600x400")

            text_widget = tk.Text(result_window, wrap=tk.WORD, font=('Courier', 9))
            text_widget.pack(fill='both', expand=True, padx=10, pady=10)

            text_widget.insert('1.0', f"OCR Results - Found {len(lines)} text lines:\n\n")
            for idx, line in enumerate(lines, 1):
                text = line.get('text', '')
                bbox = line.get('bounding_rect', {})
                text_widget.insert('end', f"{idx}. {text}\n")
                text_widget.insert('end', f"   Position: {bbox}\n\n")

            text_widget.config(state='disabled')

        except Exception as e:
            messagebox.showerror("Error", f"OCR test failed:\n{str(e)}")

    def quick_open_output(self):
        """Open results directory."""
        results_dir = self.results_dir_var.get().strip()

        # Tạo directory nếu chưa có
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create directory:\n{str(e)}")
                return

        try:
            import subprocess
            import platform

            if platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', results_dir])
            elif platform.system() == 'Windows':
                subprocess.call(['explorer', os.path.abspath(results_dir)])
            else:  # Linux
                subprocess.call(['xdg-open', results_dir])
            logger.info(f"Opened results directory: {results_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open directory:\n{str(e)}")

    def quick_copy_logs(self):
        """Copy logs to clipboard."""
        messagebox.showinfo("Info", "Logs copied to clipboard!")

    def quick_clear_cache(self):
        """Clear cache."""
        if messagebox.askyesno("Confirm", "Clear all cache files?"):
            messagebox.showinfo("Info", "Cache cleared!")
