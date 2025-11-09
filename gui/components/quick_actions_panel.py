"""
Quick Actions Panel Component
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any


class QuickActionsPanel(ttk.Frame):
    """Panel chứa các quick actions."""

    def __init__(self, parent, callbacks: Dict[str, Any]):
        super().__init__(parent, relief='solid', borderwidth=1)
        self.callbacks = callbacks

        ttk.Label(self, text="⚡ Quick Actions",
                 font=('', 11, 'bold')).pack(pady=5)

        # Buttons grid
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Row 1
        ttk.Button(
            btn_frame,
            text="📱 Check Device",
            command=lambda: self._call('check_device'),
            width=18
        ).grid(row=0, column=0, sticky='ew', padx=3, pady=3, ipady=8)

        ttk.Button(
            btn_frame,
            text="📸 Screenshot",
            command=lambda: self._call('screenshot'),
            width=18
        ).grid(row=0, column=1, sticky='ew', padx=3, pady=3, ipady=8)

        # Row 2
        ttk.Button(
            btn_frame,
            text="🔍 OCR Test",
            command=lambda: self._call('ocr_test'),
            width=18
        ).grid(row=1, column=0, sticky='ew', padx=3, pady=3, ipady=8)

        ttk.Button(
            btn_frame,
            text="📂 Open Results",
            command=lambda: self._call('open_output'),
            width=18
        ).grid(row=1, column=1, sticky='ew', padx=3, pady=3, ipady=8)

        # Row 3
        ttk.Button(
            btn_frame,
            text="📋 Copy Logs",
            command=lambda: self._call('copy_logs'),
            width=18
        ).grid(row=2, column=0, sticky='ew', padx=3, pady=3, ipady=8)

        ttk.Button(
            btn_frame,
            text="🗑 Clear Cache",
            command=lambda: self._call('clear_cache'),
            width=18
        ).grid(row=2, column=1, sticky='ew', padx=3, pady=3, ipady=8)

        # Configure grid weights
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def _call(self, action: str):
        """Call callback nếu có."""
        if action in self.callbacks:
            self.callbacks[action]()
