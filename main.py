"""
Main GUI Application
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import queue

from core.agent import Agent
from core.utils import get_logger
from gui.tabs.festival_tab import FestivalTab
from gui.tabs.gacha_tab import GachaTab
from gui.tabs.hopping_tab import HoppingTab
from gui.utils.logging_utils import OptimizedQueueHandler, OptimizedLogViewer
from gui.utils.thread_utils import get_thread_manager, shutdown_thread_manager, BackgroundTaskRunner

logger = get_logger(__name__)


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

        # Initialize Agent (without auto-connecting to device)
        try:
            self.agent = Agent(auto_connect=False)
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Agent: {e}")
            messagebox.showerror("Error", f"Cannot initialize Agent:\n{str(e)}")
            self.agent = None

        # Initialize thread manager and task runner
        thread_manager = get_thread_manager()
        self.task_runner = BackgroundTaskRunner(thread_manager)

        # Apply modern styling
        self.setup_styles()

        # Setup UI
        self.setup_ui()

        # Check initial device status
        self.check_device()

        # Start optimized log polling
        self.start_log_polling()

        logger.info("GUI initialized")

    def setup_logging(self):
        """Configure logging with buffered queue handler."""
        queue_handler = OptimizedQueueHandler(self.log_queue, buffer_size=25, flush_interval=0.3)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        queue_handler.setFormatter(formatter)
        logging.getLogger().addHandler(queue_handler)

    def setup_styles(self):
        """Configure UI styling."""
        style = ttk.Style()
        style.theme_use('alt')
        style.configure('Accent.TButton', font=('', 11, 'bold'))
        style.configure('TButton', font=('', 10))
        style.configure('TLabel', font=('', 10))
        style.configure('TEntry', font=('', 10))

    def setup_ui(self):
        """Create main user interface."""
        # Header
        header_frame = ttk.Frame(self, relief='raised', borderwidth=1)
        header_frame.pack(fill='x', padx=5, pady=5)

        left_header = ttk.Frame(header_frame)
        left_header.pack(side='left', padx=10, pady=5)
        ttk.Label(left_header, text="Auto C-Peach", font=('', 18, 'bold')).pack()
        ttk.Label(left_header, text="Game Automation Tool", font=('', 9)).pack()

        right_header = ttk.Frame(header_frame)
        right_header.pack(side='right', padx=10, pady=5)

        self.device_status_var = tk.StringVar(value="Not Connected")
        ttk.Label(right_header, textvariable=self.device_status_var, font=('', 11, 'bold')).pack()

        button_frame = ttk.Frame(right_header)
        button_frame.pack(pady=2)

        self.connect_button = ttk.Button(button_frame, text="Connect Device", command=self.connect_device,
                                        width=18, style='Accent.TButton')
        self.connect_button.pack(side='left', padx=(0, 5), ipady=5)

        ttk.Button(button_frame, text="Refresh", command=self.check_device, width=15).pack(side='left', ipady=5)

        # Main content
        content_frame = ttk.Frame(self)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Automation tabs
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill='both', expand=True, side='top')

        if self.agent:
            self.festival_tab = FestivalTab(self.notebook, self.agent)
            self.notebook.add(self.festival_tab, text="Festival Automation")

            self.gacha_tab = GachaTab(self.notebook, self.agent)
            self.notebook.add(self.gacha_tab, text="Gacha Automation")

            self.hopping_tab = HoppingTab(self.notebook, self.agent)
            self.notebook.add(self.hopping_tab, text="Hopping Automation")

        settings_tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(settings_tab, text="Settings")
        self.setup_settings_tab(settings_tab)

        # Log viewer
        self.log_viewer = OptimizedLogViewer(content_frame, self.log_queue,
                                             max_lines=int(self.max_log_lines_var.get()), poll_interval=200)

        # Footer
        footer_frame = ttk.Frame(self, relief='sunken', borderwidth=1)
        footer_frame.pack(fill='x', side='bottom')
        ttk.Label(footer_frame, text="© 2025 Auto C-Peach | Version 1.0", font=('', 8)).pack(side='left', padx=10, pady=3)
        self.footer_status_var = tk.StringVar(value="Ready")
        ttk.Label(footer_frame, textvariable=self.footer_status_var, font=('', 8)).pack(side='right', padx=10, pady=3)

    def setup_settings_tab(self, parent):
        """Create settings tab interface."""
        general_frame = ttk.LabelFrame(parent, text="General Settings", padding=15)
        general_frame.pack(fill='x', pady=10)

        log_frame = ttk.Frame(general_frame)
        log_frame.pack(fill='x', pady=5)
        ttk.Label(log_frame, text="Log Level:", font=('', 10)).pack(side='left', padx=5)

        self.log_level_var = tk.StringVar(value="INFO")
        log_combo = ttk.Combobox(log_frame, textvariable=self.log_level_var,
                                values=["DEBUG", "INFO", "WARNING", "ERROR"], state='readonly', width=15)
        log_combo.pack(side='left', padx=5)
        ttk.Button(log_frame, text="Apply", command=self.apply_log_level, width=12).pack(side='left', padx=5, ipady=5)

        theme_frame = ttk.Frame(general_frame)
        theme_frame.pack(fill='x', pady=5)
        ttk.Label(theme_frame, text="Theme:", font=('', 10)).pack(side='left', padx=5)
        self.theme_var = tk.StringVar(value="Light")
        theme_combo = ttk.Combobox(theme_frame, textvariable=self.theme_var, values=["Light", "Dark"],
                                   state='readonly', width=15)
        theme_combo.pack(side='left', padx=5)
        ttk.Label(theme_frame, text="(Coming soon)", font=('', 8)).pack(side='left', padx=5)

        perf_frame = ttk.LabelFrame(parent, text="Performance Settings", padding=15)
        perf_frame.pack(fill='x', pady=10)

        self.max_log_lines_var = tk.StringVar(value="1000")
        ttk.Label(perf_frame, text="Max Log Lines:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(perf_frame, textvariable=self.max_log_lines_var, width=10).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        ttk.Button(perf_frame, text="Apply", command=self.apply_performance_settings, width=8).grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(perf_frame, text="(Reduce for better performance)", font=('', 8)).grid(row=1, column=0, columnspan=3, sticky='w', padx=5)

        ttk.Label(perf_frame, text="Log Poll Interval (ms):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.poll_interval_var = tk.StringVar(value="200")
        ttk.Entry(perf_frame, textvariable=self.poll_interval_var, width=10).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(perf_frame, text="(50-1000ms)", font=('', 8)).grid(row=2, column=2, sticky='w', padx=5, pady=2)
        perf_frame.columnconfigure(1, weight=1)

    def apply_log_level(self):
        """Change application log level."""
        level = self.log_level_var.get()
        logging.getLogger().setLevel(level)
        logger.info(f"Log level changed to: {level}")
        messagebox.showinfo("Success", f"Log level set to: {level}")

    def apply_performance_settings(self):
        """Update performance settings."""
        try:
            max_lines = max(100, min(10000, int(self.max_log_lines_var.get())))
            poll_interval = max(50, min(1000, int(self.poll_interval_var.get())))

            self.log_viewer.set_max_lines(max_lines)
            self.log_viewer.set_poll_interval(poll_interval)

            self.max_log_lines_var.set(str(max_lines))
            self.poll_interval_var.set(str(poll_interval))

            logger.info(f"Performance settings applied: max_lines={max_lines}, poll_interval={poll_interval}ms")
            messagebox.showinfo("Success", f"Performance settings applied!\nMax lines: {max_lines}\nPoll interval: {poll_interval}ms")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid values:\n{str(e)}")

    def connect_device(self):
        """Connect to device."""
        if not self.agent:
            messagebox.showerror("Error", "Agent not initialized!")
            return

        # Store agent reference for use in nested function
        agent = self.agent
        
        self.device_status_var.set("Connecting...")
        self.footer_status_var.set("Connecting to device...")
        self.connect_button.config(state='disabled')

        def connect_task():
            try:
                success = agent.connect_device_with_retry()
                self.after(0, lambda: self._connect_finished(success))
                return success
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.after(0, lambda: self._connect_finished(False, str(e)))
                raise

        thread = get_thread_manager().submit_task("device_connection", connect_task)
        if not thread:
            self._connect_finished(False, "Failed to start connection thread")

    def _connect_finished(self, success: bool, error_msg: str = ""):
        """Handle device connection result."""
        try:
            if success:
                self.connect_button.config(state='disabled')
                self.device_status_var.set("Device Connected")
                self.footer_status_var.set("Device ready")
                logger.info("Device connected successfully")
                messagebox.showinfo("Success", "Device connected successfully!")
            else:
                self.connect_button.config(state='normal')
                self.device_status_var.set("Connection Failed")
                self.footer_status_var.set("Connection failed")
                msg = "Failed to connect to device!"
                if error_msg:
                    msg += f"\n\nError: {error_msg}"
                messagebox.showerror("Connection Failed", msg)
        except Exception as e:
            logger.error(f"Error in _connect_finished: {e}")

    def check_device(self):
        """Check current device connection status."""
        if not self.agent:
            self.device_status_var.set("Agent Error")
            self.footer_status_var.set("Agent initialization failed")
            self.connect_button.config(state='disabled')
            return

        if self.agent.is_device_connected():
            self.device_status_var.set("Device Connected")
            self.footer_status_var.set("Device ready")
            self.connect_button.config(state='disabled')
            logger.info("Device connected")
        else:
            self.device_status_var.set("Not Connected")
            self.footer_status_var.set("Device not connected")
            self.connect_button.config(state='normal')
            logger.warning("Device not connected")

    def start_log_polling(self):
        """Start log polling."""
        self.log_viewer.start_polling()

    def clear_logs(self):
        """Clear log display."""
        self.log_viewer.clear_logs()

    def save_logs(self):
        """Save logs to file."""
        self.log_viewer.save_logs()


def main():
    """Application entry point."""
    app = AutoCPeachGUI()
    try:
        app.mainloop()
    finally:
        shutdown_thread_manager()
        logger.info("Application shutdown complete")


if __name__ == '__main__':
    main()
