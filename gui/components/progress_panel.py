"""
Progress Panel Component
"""

import tkinter as tk
from tkinter import ttk
import time
from typing import Optional


class ProgressPanel(ttk.Frame):
    """Panel displaying progress and statistics."""

    def __init__(self, parent):
        super().__init__(parent, relief='solid', borderwidth=1)

        # Title
        ttk.Label(self, text="📊 Progress & Statistics",
                 font=('', 11, 'bold')).pack(pady=5)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(
            self,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress.pack(fill='x', padx=10, pady=5)

        # Progress label
        self.progress_label = ttk.Label(self, text="Ready", font=('', 9))
        self.progress_label.pack(pady=2)

        # Stats frame
        stats_frame = ttk.Frame(self)
        stats_frame.pack(fill='x', padx=10, pady=5)

        # Total
        ttk.Label(stats_frame, text="Total:", font=('', 9)).grid(row=0, column=0, sticky='w')
        self.total_label = ttk.Label(stats_frame, text="0", font=('', 9, 'bold'))
        self.total_label.grid(row=0, column=1, sticky='w', padx=5)

        # Success
        ttk.Label(stats_frame, text=" OK:", font=('', 9)).grid(row=0, column=2, sticky='w', padx=(20, 0))
        self.ok_label = ttk.Label(stats_frame, text="0", font=('', 9, 'bold'), foreground='green')
        self.ok_label.grid(row=0, column=3, sticky='w', padx=5)

        # Failed
        ttk.Label(stats_frame, text=" NG:", font=('', 9)).grid(row=0, column=4, sticky='w', padx=(20, 0))
        self.ng_label = ttk.Label(stats_frame, text="0", font=('', 9, 'bold'), foreground='red')
        self.ng_label.grid(row=0, column=5, sticky='w', padx=5)

        # Time elapsed
        ttk.Label(stats_frame, text="⏱ Time:", font=('', 9)).grid(row=1, column=0, sticky='w', pady=(5, 0))
        self.time_label = ttk.Label(stats_frame, text="00:00:00", font=('', 9, 'bold'))
        self.time_label.grid(row=1, column=1, columnspan=2, sticky='w', padx=5, pady=(5, 0))

        # ETA
        ttk.Label(stats_frame, text="⏳ ETA:", font=('', 9)).grid(row=1, column=2, sticky='w', padx=(20, 0), pady=(5, 0))
        self.eta_label = ttk.Label(stats_frame, text="--:--:--", font=('', 9, 'bold'))
        self.eta_label.grid(row=1, column=3, columnspan=3, sticky='w', padx=5, pady=(5, 0))

        # Initialize counters
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.total = 0
        self.ok_count = 0
        self.ng_count = 0
        self.start_time = None
        self.update_display()

    def start(self, total: int):
        """Start progress tracking."""
        self.total = total
        self.ok_count = 0
        self.ng_count = 0
        self.start_time = time.time()
        self.update_display()

    def update_result(self, ok: bool):
        """Update progress with new result."""
        if ok:
            self.ok_count += 1
        else:
            self.ng_count += 1
        if (self.ok_count + self.ng_count) % 5 == 0:
            self.update_display()

    def update_display(self):
        """Update progress display."""
        self.total_label.config(text=str(self.total))
        self.ok_label.config(text=str(self.ok_count))
        self.ng_label.config(text=str(self.ng_count))

        if self.total > 0:
            processed = self.ok_count + self.ng_count
            progress = (processed / self.total) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"{processed}/{self.total}")
        else:
            self.progress_var.set(0)
            self.progress_label.config(text="Ready")

        if self.start_time:
            elapsed = time.time() - self.start_time
            self.time_label.config(text=self._format_time(elapsed))

            processed = self.ok_count + self.ng_count
            if 0 < processed < self.total:
                remaining = elapsed / processed * (self.total - processed)
                self.eta_label.config(text=self._format_time(remaining))
            elif processed >= self.total:
                self.eta_label.config(text="Done")

    def _format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
