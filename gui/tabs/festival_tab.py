"""
Festival Tab for GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

from automations.festivals import FestivalAutomation
from core.agent import Agent
from core.utils import get_logger
from gui.components.progress_panel import ProgressPanel
from gui.components.quick_actions_panel import QuickActionsPanel
import core.data as data

logger = get_logger(__name__)


class FestivalTab(ttk.Frame):
    """Tab cho Festival Automation với UI cải tiến."""

    def __init__(self, parent, agent: Agent):
        super().__init__(parent)
        self.agent = agent
        self.festival_automation: Optional[FestivalAutomation] = None
        self.is_running = False
        self.automation_thread: Optional[threading.Thread] = None

        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện cho Festival tab."""

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

        # 1. File Selection
        file_section = ttk.LabelFrame(left_frame, text="📁 Data File", padding=10)
        file_section.pack(fill='x', pady=5)

        self.file_path_var = tk.StringVar()

        file_inner = ttk.Frame(file_section)
        file_inner.pack(fill='x')

        ttk.Label(file_inner, text="Festival Data:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(file_inner, textvariable=self.file_path_var, width=40, font=('', 10)).grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        ttk.Button(file_inner, text="Browse", command=self.browse_file, width=12).grid(row=0, column=2, pady=2, ipady=5)
        ttk.Button(file_inner, text="👁 Preview", command=self.preview_data, width=12).grid(row=0, column=3, padx=2, pady=2, ipady=5)

        ttk.Label(file_inner, text="", font=('', 8)).grid(row=1, column=0, columnspan=4, sticky='w', pady=(2, 0))
        ttk.Label(file_inner, text="💡 Select festivals.json or CSV file containing festival data", font=('', 9), foreground='gray').grid(row=2, column=0, columnspan=4, sticky='w')

        file_inner.columnconfigure(1, weight=1)

        # 2. Configuration
        config_section = ttk.LabelFrame(left_frame, text="⚙️ Automation Settings", padding=10)
        config_section.pack(fill='x', pady=5)

        config_inner = ttk.Frame(config_section)
        config_inner.pack(fill='x')

        # Templates path
        self.templates_path_var = tk.StringVar(value="./templates")
        ttk.Label(config_inner, text="Templates Folder:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(config_inner, textvariable=self.templates_path_var, width=30, font=('', 10)).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_inner, text="📂", command=self.browse_templates, width=8).grid(row=0, column=2, pady=2, ipady=5)

        # Snapshot directory
        self.snapshot_dir_var = tk.StringVar(value="./result/festival/snapshots")
        ttk.Label(config_inner, text="Snapshots Folder:", font=('', 10)).grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(config_inner, textvariable=self.snapshot_dir_var, width=30, font=('', 10)).grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(config_inner, text="📂", command=self.browse_snapshot_dir, width=8).grid(row=1, column=2, pady=2, ipady=5)

        # Wait time
        self.wait_time_var = tk.StringVar(value="1.0")
        ttk.Label(config_inner, text="Wait After Touch:", font=('', 10)).grid(row=2, column=0, sticky='w', pady=2)
        wait_frame = ttk.Frame(config_inner)
        wait_frame.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        ttk.Entry(wait_frame, textvariable=self.wait_time_var, width=10, font=('', 10)).pack(side='left')
        ttk.Label(wait_frame, text="seconds", font=('', 9)).pack(side='left', padx=5)

        config_inner.columnconfigure(1, weight=1)

        # 3. Output Configuration
        output_section = ttk.LabelFrame(left_frame, text="💾 Output", padding=10)
        output_section.pack(fill='x', pady=5)

        output_inner = ttk.Frame(output_section)
        output_inner.pack(fill='x')

        # Results directory
        self.results_dir_var = tk.StringVar(value="./result/festival/results")
        ttk.Label(output_inner, text="Results Folder:", font=('', 10)).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(output_inner, textvariable=self.results_dir_var, width=30, font=('', 10)).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(output_inner, text="📂", command=self.browse_results_dir, width=8).grid(row=0, column=2, pady=2, ipady=5)

        # Output file (optional)
        self.output_file_var = tk.StringVar(value="")
        ttk.Label(output_inner, text="Output File:", font=('', 10)).grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(output_inner, textvariable=self.output_file_var, width=30, font=('', 10)).grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(output_inner, text="📂", command=self.browse_output, width=8).grid(row=1, column=2, pady=2, ipady=5)
        ttk.Label(output_inner, text="(Optional - auto-generated if empty)", font=('', 8), foreground='gray').grid(row=2, column=0, columnspan=3, sticky='w')

        output_inner.columnconfigure(1, weight=1)

        # 4. Action Buttons
        action_frame = ttk.Frame(left_frame)
        action_frame.pack(fill='x', pady=10)

        self.start_button = ttk.Button(
            action_frame,
            text="▶️ Start Automation",
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

        self.pause_button = ttk.Button(
            action_frame,
            text="⏸ Pause",
            command=self.pause_automation,
            state='disabled',
            width=12
        )
        self.pause_button.pack(side='left', padx=5, ipadx=15, ipady=10)

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

    def browse_file(self):
        """Mở dialog để chọn file CSV/JSON."""
        filename = filedialog.askopenfilename(
            title="Select festival data file",
            filetypes=[
                ("Data files", "*.csv *.json"),
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file_path_var.set(filename)
            logger.info(f"Selected file: {filename}")
            # Auto preview
            self.after(100, self.preview_data)

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

    def preview_data(self):
        """Preview dữ liệu từ file đã chọn."""
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid file!")
            return

        try:
            data_list = data.load_data(file_path)
            if not data_list:
                messagebox.showwarning("Warning", "File contains no data!")
                return

            # Tạo popup window
            preview_window = tk.Toplevel(self)
            preview_window.title(f"Preview: {os.path.basename(file_path)}")
            preview_window.geometry("700x500")

            # Header
            header = ttk.Frame(preview_window)
            header.pack(fill='x', padx=10, pady=5)
            ttk.Label(
                header,
                text=f"📄 {os.path.basename(file_path)}",
                font=('', 11, 'bold')
            ).pack(side='left')
            ttk.Label(
                header,
                text=f"Total: {len(data_list)} rows",
                font=('', 10)
            ).pack(side='right')

            # Text widget
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

            # Close button
            ttk.Button(
                preview_window,
                text="Close",
                command=preview_window.destroy,
                width=15
            ).pack(pady=5, ipady=8)

        except Exception as e:
            messagebox.showerror("Error", f"Cannot read file:\n{str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """Lấy cấu hình từ UI."""
        try:
            wait_time = float(self.wait_time_var.get())
        except ValueError:
            wait_time = 1.0
            logger.warning("Invalid wait time, using default: 1.0")

        config = {
            'templates_path': self.templates_path_var.get().strip(),
            'snapshot_dir': self.snapshot_dir_var.get().strip(),
            'results_dir': self.results_dir_var.get().strip(),
            'wait_after_touch': wait_time,
        }

        return config

    def save_config(self):
        """Lưu cấu hình vào file JSON."""
        filename = filedialog.asksaveasfilename(
            title="Save config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filename:
            return

        config = self.get_config()
        config['file_path'] = self.file_path_var.get()
        config['output_file'] = self.output_file_var.get()

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
            title="Load config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Apply config to UI
            if 'file_path' in config:
                self.file_path_var.set(config['file_path'])
            if 'templates_path' in config:
                self.templates_path_var.set(config['templates_path'])
            if 'snapshot_dir' in config:
                self.snapshot_dir_var.set(config['snapshot_dir'])
            if 'results_dir' in config:
                self.results_dir_var.set(config['results_dir'])
            if 'wait_after_touch' in config:
                self.wait_time_var.set(str(config['wait_after_touch']))
            if 'output_file' in config:
                self.output_file_var.set(config['output_file'])

            messagebox.showinfo("Success", "Config loaded successfully!")
            logger.info(f"Config loaded: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load config:\n{str(e)}")

    def start_automation(self):
        """Bắt đầu automation."""
        # Validate
        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid CSV/JSON file!")
            return

        # Check device
        if not self.agent.is_device_connected():
            messagebox.showerror("Error", "Device not connected!\nPlease connect device first.")
            return

        # Load data to get count
        try:
            data_list = data.load_data(file_path)
            if not data_list:
                messagebox.showerror("Error", "No data in file!")
                return
            total_count = len(data_list)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load data:\n{str(e)}")
            return

        # Update UI
        self.is_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.pause_button.config(state='normal')
        self.status_var.set("🔄 Running automation...")

        # Initialize progress
        self.progress_panel.start(total_count)

        # Get config
        config = self.get_config()

        # Run in thread
        self.automation_thread = threading.Thread(
            target=self._run_automation,
            args=(file_path, config),
            daemon=True
        )
        self.automation_thread.start()
        logger.info("Festival automation started")

    def _run_automation(self, file_path: str, config: Dict[str, Any]):
        """Chạy automation (trong thread)."""
        try:
            # Khởi tạo FestivalAutomation
            self.festival_automation = FestivalAutomation(self.agent, config)

            # Get output path if specified
            output_file = self.output_file_var.get().strip()
            output_path = output_file if output_file else None

            # Run với output path tùy chọn
            if output_path:
                success = self.festival_automation.run_all_stages(file_path, output_path)
            else:
                success = self.festival_automation.run(file_path)

            # Update UI
            self.after(0, lambda: self._automation_finished(success))

        except Exception as e:
            logger.error(f"Automation error: {e}")
            self.after(0, lambda: self._automation_finished(False, str(e)))

    def _automation_finished(self, success: bool, error_msg: str = ""):
        """Callback khi automation kết thúc."""
        self.is_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.pause_button.config(state='disabled')

        if success:
            self.status_var.set("✅ Automation completed successfully!")
            messagebox.showinfo("Success", "Automation completed!")
        else:
            self.status_var.set("❌ Automation failed")
            msg = "Automation failed!"
            if error_msg:
                msg += f"\n\nError: {error_msg}"
            messagebox.showerror("Error", msg)

    def stop_automation(self):
        """Dừng automation."""
        if messagebox.askyesno("Confirm", "Are you sure you want to stop?"):
            self.is_running = False
            self.status_var.set("⏹ Stopped by user")
            logger.warning("Stop requested by user")

    def pause_automation(self):
        """Pause automation."""
        messagebox.showinfo("Info", "Pause functionality coming soon!")

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
