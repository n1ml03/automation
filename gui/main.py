"""
Main GUI Application
"""

import sys
import os
# Add parent directory to path so we can import core module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import logging
import queue
from typing import Optional

from core.agent import Agent
from core.utils import get_logger
from gui.tabs.festival_tab import FestivalTab
from gui.tabs.gacha_tab import GachaTab
from gui.tabs.hopping_tab import HoppingTab

logger = get_logger(__name__)


class QueueHandler(logging.Handler):
    """Custom logging handler để gửi log vào queue cho GUI."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        log_entry = self.format(record)
        self.log_queue.put(log_entry)


class AutoCPeachGUI(tk.Tk):
    """Main GUI Application"""

    def __init__(self):
        super().__init__()

        self.title("Auto C-Peach")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Setup logging queue
        self.log_queue = queue.Queue()
        self.setup_logging()

        # Initialize Agent
        try:
            self.agent = Agent()
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Agent: {e}")
            messagebox.showerror("Error", f"Cannot initialize Agent:\n{str(e)}")
            self.agent = None

        # Apply modern styling
        self.setup_styles()

        # Setup UI
        self.setup_ui()

        # Start log polling
        self.start_log_polling()

        logger.info("GUI initialized")

    def setup_logging(self):
        """Setup logging."""
        queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                     datefmt='%H:%M:%S')
        queue_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(queue_handler)

    def setup_styles(self):
        """Setup modern styling."""
        style = ttk.Style()

        # Use modern theme
        style.theme_use('alt')

        # Configure colors and button styles
        style.configure('Accent.TButton', font=('', 11, 'bold'))
        style.configure('TButton', font=('', 10))
        style.configure('TLabel', font=('', 10))
        style.configure('TEntry', font=('', 10))

    def setup_ui(self):
        """Thiết lập giao diện chính."""

        # === HEADER ===
        header_frame = ttk.Frame(self, relief='raised', borderwidth=1)
        header_frame.pack(fill='x', padx=5, pady=5)

        # Left side - Title
        left_header = ttk.Frame(header_frame)
        left_header.pack(side='left', padx=10, pady=5)

        title_label = ttk.Label(
            left_header,
            text="🎮 Auto C-Peach",
            font=('', 18, 'bold')
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            left_header,
            text="Game Automation Tool",
            font=('', 9)
        )
        subtitle_label.pack()

        # Right side - Device status
        right_header = ttk.Frame(header_frame)
        right_header.pack(side='right', padx=10, pady=5)

        self.device_status_var = tk.StringVar(value="❓ Unknown")
        status_label = ttk.Label(
            right_header,
            textvariable=self.device_status_var,
            font=('', 11, 'bold')
        )
        status_label.pack()

        ttk.Button(
            right_header,
            text="🔄 Refresh",
            command=self.check_device,
            width=15
        ).pack(pady=2, ipady=5)

        # === MAIN CONTENT ===
        content_frame = ttk.Frame(self)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill='both', expand=True, side='top')

        # Tab 1: Festival Automation
        if self.agent:
            self.festival_tab = FestivalTab(self.notebook, self.agent)
            self.notebook.add(self.festival_tab, text="🎪 Festival Automation")

        # Tab 2: Gacha Automation
        if self.agent:
            self.gacha_tab = GachaTab(self.notebook, self.agent)
            self.notebook.add(self.gacha_tab, text="🎰 Gacha Automation")

        # Tab 3: Hopping Automation
        if self.agent:
            self.hopping_tab = HoppingTab(self.notebook, self.agent)
            self.notebook.add(self.hopping_tab, text="🌍 Hopping Automation")

        # Tab 4: Settings
        settings_tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(settings_tab, text="⚙️ Settings")
        self.setup_settings_tab(settings_tab)

        # === LOG VIEWER ===
        log_container = ttk.Frame(content_frame)
        log_container.pack(fill='both', expand=True, side='bottom', pady=(5, 0))

        # Log header with controls
        log_header = ttk.Frame(log_container)
        log_header.pack(fill='x')

        ttk.Label(log_header, text="📋 Activity Logs", font=('', 10, 'bold')).pack(side='left', padx=5)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            log_header,
            text="Auto-scroll",
            variable=self.auto_scroll_var
        ).pack(side='right', padx=5)

        ttk.Button(
            log_header,
            text="🗑 Clear",
            command=self.clear_logs,
            width=10
        ).pack(side='right', padx=5, ipady=4)

        ttk.Button(
            log_header,
            text="💾 Save",
            command=self.save_logs,
            width=10
        ).pack(side='right', padx=5, ipady=4)

        # Log text widget
        log_frame = ttk.Frame(log_container)
        log_frame.pack(fill='both', expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            wrap=tk.WORD,
            font=('Courier', 9)
        )
        self.log_text.pack(fill='both', expand=True)

        # === FOOTER ===
        footer_frame = ttk.Frame(self, relief='sunken', borderwidth=1)
        footer_frame.pack(fill='x', side='bottom')

        ttk.Label(
            footer_frame,
            text="© 2025 Auto C-Peach | Version 1.0",
            font=('', 8)
        ).pack(side='left', padx=10, pady=3)

        self.footer_status_var = tk.StringVar(value="Ready")
        ttk.Label(
            footer_frame,
            textvariable=self.footer_status_var,
            font=('', 8)
        ).pack(side='right', padx=10, pady=3)

        # Initial device check
        self.after(500, self.check_device)

    def setup_settings_tab(self, parent):
        """Setup Settings tab."""

        # General Settings Section
        general_frame = ttk.LabelFrame(parent, text="⚙️ General Settings", padding=15)
        general_frame.pack(fill='x', pady=10)

        # Log level
        log_frame = ttk.Frame(general_frame)
        log_frame.pack(fill='x', pady=5)

        ttk.Label(log_frame, text="Log Level:", font=('', 10)).pack(side='left', padx=5)

        self.log_level_var = tk.StringVar(value="INFO")
        log_combo = ttk.Combobox(
            log_frame,
            textvariable=self.log_level_var,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            state='readonly',
            width=15
        )
        log_combo.pack(side='left', padx=5)

        ttk.Button(
            log_frame,
            text="Apply",
            command=self.apply_log_level,
            width=12
        ).pack(side='left', padx=5, ipady=5)

        # Theme (placeholder)
        theme_frame = ttk.Frame(general_frame)
        theme_frame.pack(fill='x', pady=5)

        ttk.Label(theme_frame, text="Theme:", font=('', 10)).pack(side='left', padx=5)

        self.theme_var = tk.StringVar(value="Light")
        theme_combo = ttk.Combobox(
            theme_frame,
            textvariable=self.theme_var,
            values=["Light", "Dark"],
            state='readonly',
            width=15
        )
        theme_combo.pack(side='left', padx=5)

        ttk.Label(theme_frame, text="(Coming soon)", font=('', 8)).pack(side='left', padx=5)

        # Performance Settings Section
        perf_frame = ttk.LabelFrame(parent, text="⚡ Performance Settings", padding=15)
        perf_frame.pack(fill='x', pady=10)

        self.max_log_lines_var = tk.StringVar(value="1000")

        ttk.Label(perf_frame, text="Max Log Lines:").pack(side='left', padx=5)
        ttk.Entry(perf_frame, textvariable=self.max_log_lines_var, width=10).pack(side='left', padx=5)
        ttk.Label(perf_frame, text="(Reduce for better performance)", font=('', 8)).pack(side='left', padx=5)

    def apply_log_level(self):
        """Áp dụng log level."""
        level = self.log_level_var.get()
        logging.getLogger().setLevel(level)
        logger.info(f"Log level changed to: {level}")
        messagebox.showinfo("Success", f"Log level set to: {level}")

    def check_device(self):
        """Kiểm tra device connection."""
        if not self.agent:
            self.device_status_var.set("❌ Agent Error")
            self.footer_status_var.set("Agent initialization failed")
            return

        if self.agent.is_device_connected():
            self.device_status_var.set("✅ Device Connected")
            self.footer_status_var.set("Device ready")
            logger.info("Device connected")
        else:
            self.device_status_var.set("❌ Not Connected")
            self.footer_status_var.set("Device not connected")
            logger.warning("Device not connected")

    def start_log_polling(self):
        """Start polling logs."""
        self.poll_logs()

    def poll_logs(self):
        """Poll log queue và update UI."""
        updated = False
        while True:
            try:
                log_entry = self.log_queue.get_nowait()
                self.log_text.insert('end', log_entry + '\n')
                updated = True

                # Limit log lines for performance
                try:
                    max_lines = int(self.max_log_lines_var.get())
                    current_lines = int(self.log_text.index('end-1c').split('.')[0])
                    if current_lines > max_lines:
                        self.log_text.delete('1.0', f'{current_lines - max_lines}.0')
                except:
                    pass

            except queue.Empty:
                break

        # Auto scroll if enabled
        if updated and self.auto_scroll_var.get():
            self.log_text.see('end')

        # Schedule next poll
        self.after(100, self.poll_logs)

    def clear_logs(self):
        """Clear logs."""
        self.log_text.delete('1.0', 'end')
        logger.info("Logs cleared")

    def save_logs(self):
        """Save logs to file."""
        filename = filedialog.asksaveasfilename(
            title="Save logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get('1.0', 'end'))
                messagebox.showinfo("Success", f"Logs saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot save logs:\n{str(e)}")


def main():
    """Entry point."""
    app = AutoCPeachGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
