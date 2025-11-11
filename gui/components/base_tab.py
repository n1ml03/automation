"""
Base Tab Class for Automation Tabs
Eliminates code duplication across Festival/Gacha/Hopping tabs
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
from datetime import datetime
from typing import Dict, Any, Optional
from core.agent import Agent
from core.utils import get_logger
from core.config import DEFAULT_PATHS
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
        self.templates_path_var = tk.StringVar(value=DEFAULT_PATHS['templates'])
        self.snapshot_dir_var = tk.StringVar(value=f"{DEFAULT_PATHS['results']}/{tab_name.lower()}/snapshots")
        self.results_dir_var = tk.StringVar(value=f"{DEFAULT_PATHS['results']}/{tab_name.lower()}/results")
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
        """Preview data from selected file with enhanced table view and performance optimization."""
        from core.data import load_data

        file_path = self.file_path_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid file!")
            return

        try:
            data_list = load_data(file_path)
            if not data_list:
                messagebox.showwarning("Warning", "File contains no data!")
                return

            # Create preview window
            preview_window = tk.Toplevel(self)
            preview_window.title(f"📊 Data Preview: {os.path.basename(file_path)}")
            preview_window.geometry("1200x700")
            preview_window.minsize(800, 500)
            
            # Get all unique columns from all rows
            all_columns = []
            for row in data_list:
                for key in row.keys():
                    if key not in all_columns:
                        all_columns.append(key)
            
            # Performance optimization: Use lazy loading for large datasets
            MAX_DISPLAY_ROWS = 1000
            is_large_dataset = len(data_list) > MAX_DISPLAY_ROWS
            display_data = data_list[:MAX_DISPLAY_ROWS] if is_large_dataset else data_list
            
            # ===== TOP TOOLBAR =====
            toolbar = ttk.Frame(preview_window, relief='groove', borderwidth=1)
            toolbar.pack(fill='x', padx=5, pady=5)
            
            # File info section
            info_container = ttk.Frame(toolbar)
            info_container.pack(side='left', fill='x', expand=True, padx=10, pady=8)
            
            ttk.Label(info_container, text=f"📄 {os.path.basename(file_path)}", 
                     font=('Segoe UI', 10, 'bold')).pack(side='left')
            ttk.Label(info_container, text=f"  •  {len(data_list):,} rows", 
                     font=('', 9), foreground='#2563eb').pack(side='left', padx=8)
            ttk.Label(info_container, text=f"  •  {len(all_columns)} columns", 
                     font=('', 9), foreground='#059669').pack(side='left')
            
            if is_large_dataset:
                ttk.Label(info_container, text=f"  ⚡ Showing first {MAX_DISPLAY_ROWS:,} rows", 
                         font=('', 9), foreground='#dc2626').pack(side='left', padx=8)
            
            # Search section
            search_container = ttk.Frame(toolbar)
            search_container.pack(side='right', padx=10, pady=5)
            
            ttk.Label(search_container, text="🔍", font=('', 11)).pack(side='left', padx=(0, 5))
            search_var = tk.StringVar()
            search_entry = ttk.Entry(search_container, textvariable=search_var, width=30, font=('', 9))
            search_entry.pack(side='left', padx=2)
            
            clear_search_btn = ttk.Button(search_container, text="✕", width=3, 
                                         command=lambda: search_var.set(""))
            clear_search_btn.pack(side='left', padx=2)
            
            # ===== SIDE PANEL FOR COLUMN CONTROL =====
            main_container = ttk.Frame(preview_window)
            main_container.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Left panel - Column visibility
            left_panel = ttk.LabelFrame(main_container, text="📊 Columns", width=200)
            left_panel.pack(side='left', fill='y', padx=(0, 5))
            left_panel.pack_propagate(False)
            
            # Column visibility controls
            col_control_frame = ttk.Frame(left_panel)
            col_control_frame.pack(fill='x', padx=5, pady=5)
            
            ttk.Button(col_control_frame, text="✓ All", width=8,
                      command=lambda: toggle_all_columns(True)).pack(side='left', padx=2)
            ttk.Button(col_control_frame, text="✗ None", width=8,
                      command=lambda: toggle_all_columns(False)).pack(side='left', padx=2)
            
            # Scrollable column list
            col_canvas = tk.Canvas(left_panel, highlightthickness=0)
            col_scrollbar = ttk.Scrollbar(left_panel, orient='vertical', command=col_canvas.yview)
            col_list_frame = ttk.Frame(col_canvas)
            
            col_canvas.pack(side='left', fill='both', expand=True, padx=5, pady=5)
            col_scrollbar.pack(side='right', fill='y')
            col_canvas.configure(yscrollcommand=col_scrollbar.set)
            
            col_canvas_window = col_canvas.create_window((0, 0), window=col_list_frame, anchor='nw')
            
            # Column checkboxes
            column_vars = {}
            for col in all_columns:
                var = tk.BooleanVar(value=True)
                column_vars[col] = var
                cb = ttk.Checkbutton(col_list_frame, text=col, variable=var,
                                    command=lambda c=col: toggle_column_visibility(c))
                cb.pack(anchor='w', padx=5, pady=2)
            
            col_list_frame.bind('<Configure>', lambda e: col_canvas.configure(scrollregion=col_canvas.bbox('all')))
            col_canvas.bind('<Configure>', lambda e: col_canvas.itemconfig(col_canvas_window, width=e.width))
            
            # Statistics panel
            stats_frame = ttk.LabelFrame(left_panel, text="📈 Statistics")
            stats_frame.pack(fill='x', padx=5, pady=5, side='bottom')
            
            stats_text = tk.Text(stats_frame, height=8, width=25, font=('Courier', 8), wrap=tk.WORD)
            stats_text.pack(fill='x', padx=5, pady=5)
            
            def update_stats():
                """Update statistics display."""
                stats_text.delete('1.0', 'end')
                visible_items = tree.get_children()
                stats_text.insert('end', f"Visible: {len(visible_items)}/{len(data_list)}\n")
                stats_text.insert('end', f"Selected: {len(tree.selection())}\n")
                stats_text.insert('end', f"Columns: {len(all_columns)}\n")
                stats_text.config(state='disabled')
            
            # ===== RIGHT PANEL - TABLE =====
            right_panel = ttk.Frame(main_container)
            right_panel.pack(side='right', fill='both', expand=True)
            
            # Table container with scrollbars
            table_frame = ttk.Frame(right_panel)
            table_frame.pack(fill='both', expand=True)
            
            tree_scroll_y = ttk.Scrollbar(table_frame, orient='vertical')
            tree_scroll_y.pack(side='right', fill='y')
            
            tree_scroll_x = ttk.Scrollbar(table_frame, orient='horizontal')
            tree_scroll_x.pack(side='bottom', fill='x')
            
            # Create Treeview with enhanced styling
            tree = ttk.Treeview(
                table_frame,
                columns=all_columns,
                show='tree headings',
                yscrollcommand=tree_scroll_y.set,
                xscrollcommand=tree_scroll_x.set,
                selectmode='extended',
                height=20
            )
            tree.pack(side='left', fill='both', expand=True)
            
            tree_scroll_y.config(command=tree.yview)
            tree_scroll_x.config(command=tree.xview)
            
            # Configure row tags for alternating colors
            tree.tag_configure('oddrow', background='#f8fafc')
            tree.tag_configure('evenrow', background='#ffffff')
            tree.tag_configure('selected', background='#dbeafe')
            
            # Configure columns with smart width
            tree.column('#0', width=70, minwidth=50, anchor='center')
            tree.heading('#0', text='#', anchor='center')
            
            visible_columns = set(all_columns)
            
            for col in all_columns:
                # Smart column width calculation
                max_width = len(str(col)) * 9
                sample_size = min(100, len(display_data))
                for row in display_data[:sample_size]:
                    value = str(row.get(col, ''))
                    max_width = max(max_width, min(len(value) * 7, 300))
                
                max_width = min(max_width, 350)
                max_width = max(max_width, 80)
                
                tree.column(col, width=max_width, minwidth=60, anchor='w', stretch=True)
                tree.heading(col, text=f"▼ {col}", anchor='w')
            
            # Performance optimized data loading
            original_data = []
            for idx, row in enumerate(display_data, 1):
                values = [row.get(col, '') for col in all_columns]
                tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
                item_id = tree.insert('', 'end', text=str(idx), values=values, tags=(tag,))
                original_data.append((item_id, idx, row, values))
            
            # Search with performance optimization
            search_job = None
            def search_data(*args):
                nonlocal search_job
                if search_job:
                    preview_window.after_cancel(search_job)
                search_job = preview_window.after(300, perform_search)
            
            def perform_search():
                search_text = search_var.get().lower().strip()
                
                # Performance optimization: detach/reattach is faster than delete/insert
                if not search_text:
                    for item_id, idx, row, values in original_data:
                        if not tree.exists(item_id):
                            continue
                        tree.reattach(item_id, '', 'end')
                else:
                    for item_id, idx, row, values in original_data:
                        if not tree.exists(item_id):
                            continue
                        matches = any(search_text in str(value).lower() for value in row.values())
                        if matches:
                            tree.reattach(item_id, '', 'end')
                        else:
                            tree.detach(item_id)
                
                update_stats()
            
            search_var.trace('w', search_data)
            
            # Enhanced column sorting with visual indicators
            sort_state = {}  # Track sort state per column
            
            def sort_column(col, reverse):
                """Sort treeview by column with visual feedback."""
                items = [(tree.set(item, col), item) for item in tree.get_children('')]
                
                # Try numeric sort first, fall back to string sort
                try:
                    items.sort(key=lambda x: float(str(x[0])) if x[0] else 0, reverse=reverse)
                except (ValueError, TypeError):
                    items.sort(key=lambda x: str(x[0]).lower(), reverse=reverse)
                
                for index, (val, item) in enumerate(items):
                    tree.move(item, '', index)
                
                # Update heading with sort indicator
                indicator = "▲" if reverse else "▼"
                tree.heading(col, text=f"{indicator} {col}", 
                           command=lambda: sort_column(col, not reverse))
                
                # Clear other column indicators
                for other_col in all_columns:
                    if other_col != col:
                        tree.heading(other_col, text=f"▼ {other_col}")
                
                sort_state[col] = reverse
            
            # Add sort to each column
            for col in all_columns:
                tree.heading(col, command=lambda c=col: sort_column(c, False))
            
            # Column visibility toggle
            def toggle_column_visibility(col):
                if column_vars[col].get():
                    tree['displaycolumns'] = [c for c in all_columns if column_vars[c].get()]
                else:
                    tree['displaycolumns'] = [c for c in all_columns if column_vars[c].get()]
            
            def toggle_all_columns(show):
                for col in all_columns:
                    column_vars[col].set(show)
                tree['displaycolumns'] = all_columns if show else []
            
            # Context menu (right-click)
            context_menu = tk.Menu(preview_window, tearoff=0)
            context_menu.add_command(label="📋 Copy Cell", command=lambda: copy_selected_cell())
            context_menu.add_command(label="📋 Copy Row", command=lambda: copy_selected_row())
            context_menu.add_separator()
            context_menu.add_command(label="🔍 Filter by Value", command=lambda: filter_by_value())
            context_menu.add_separator()
            context_menu.add_command(label="✓ Select All", command=lambda: tree.selection_set(tree.get_children()))
            
            def show_context_menu(event):
                try:
                    context_menu.tk_popup(event.x_root, event.y_root)
                finally:
                    context_menu.grab_release()
            
            tree.bind('<Button-3>', show_context_menu)
            
            # Copy functions
            def copy_selected_cell():
                item = tree.focus()
                if item:
                    selection = tree.selection()
                    if selection:
                        item = selection[0]
                        values = tree.item(item, 'values')
                        if values:
                            preview_window.clipboard_clear()
                            preview_window.clipboard_append(str(values[0]))
            
            def copy_selected_row():
                selection = tree.selection()
                if selection:
                    rows = []
                    for item in selection:
                        row_num = tree.item(item, 'text')
                        values = tree.item(item, 'values')
                        rows.append('\t'.join([row_num] + [str(v) for v in values]))
                    preview_window.clipboard_clear()
                    preview_window.clipboard_append('\n'.join(rows))
            
            def filter_by_value():
                item = tree.focus()
                if item:
                    values = tree.item(item, 'values')
                    if values:
                        search_var.set(str(values[0]))
            
            # Double-click to copy cell
            def copy_cell_double_click(event):
                item = tree.focus()
                if item:
                    col = tree.identify_column(event.x)
                    if col == '#0':
                        value = tree.item(item, 'text')
                    else:
                        col_index = int(col.replace('#', '')) - 1
                        value = tree.item(item, 'values')[col_index]
                    preview_window.clipboard_clear()
                    preview_window.clipboard_append(str(value))
                    # Visual feedback
                    status_label.config(text=f"✓ Copied: {str(value)[:50]}...", foreground='#059669')
                    preview_window.after(2000, lambda: status_label.config(text="Ready", foreground='#64748b'))
            
            tree.bind('<Double-Button-1>', copy_cell_double_click)
            
            # Keyboard shortcuts
            def on_key_press(event):
                if event.state & 0x4:  # Ctrl key
                    if event.keysym == 'f':
                        search_entry.focus()
                        search_entry.select_range(0, 'end')
                        return 'break'
                    elif event.keysym == 'c':
                        copy_selected_row()
                        return 'break'
                    elif event.keysym == 'a':
                        tree.selection_set(tree.get_children())
                        return 'break'
                elif event.keysym == 'Escape':
                    preview_window.destroy()
                    return 'break'
            
            preview_window.bind('<Key>', on_key_press)
            
            # Selection change handler
            def on_selection_change(event):
                update_stats()
            
            tree.bind('<<TreeviewSelect>>', on_selection_change)
            
            # ===== STATUS BAR =====
            status_bar = ttk.Frame(right_panel, relief='sunken', borderwidth=1)
            status_bar.pack(fill='x', pady=(5, 0))
            
            status_label = ttk.Label(status_bar, text="Ready", font=('', 9), foreground='#64748b')
            status_label.pack(side='left', padx=10, pady=3)
            
            ttk.Label(status_bar, text="💡 Ctrl+F: Search  •  Ctrl+C: Copy  •  Right-click: Menu  •  ESC: Close", 
                     font=('', 8), foreground='#94a3b8').pack(side='right', padx=10)
            
            # ===== BOTTOM BUTTONS =====
            button_frame = ttk.Frame(preview_window)
            button_frame.pack(fill='x', padx=10, pady=8)
            
            ttk.Button(button_frame, text="📋 Copy All", 
                      command=lambda: self._copy_table_data(tree, all_columns), 
                      width=14).pack(side='left', padx=3)
            ttk.Button(button_frame, text="💾 Export CSV", 
                      command=lambda: self._export_table_data(data_list, all_columns), 
                      width=14).pack(side='left', padx=3)
            ttk.Button(button_frame, text="📊 Column Stats", 
                      command=lambda: self._show_column_stats(data_list, all_columns), 
                      width=14).pack(side='left', padx=3)
            ttk.Button(button_frame, text="🔄 Refresh", 
                      command=lambda: messagebox.showinfo("Refresh", "Data refreshed!"), 
                      width=12).pack(side='left', padx=3)
            
            ttk.Button(button_frame, text="Close", 
                      command=preview_window.destroy, 
                      width=12).pack(side='right', padx=3, ipady=3)
            
            # Initialize stats
            update_stats()
            
            # Focus on search entry
            preview_window.after(100, lambda: search_entry.focus())

        except Exception as e:
            logger.exception(f"Data preview error: {e}")
            messagebox.showerror("Error", f"Cannot read file:\n{str(e)}")
    
    def _copy_table_data(self, tree, columns):
        """Copy table data to clipboard."""
        try:
            # Header
            data = '\t'.join(['#'] + list(columns)) + '\n'
            
            # Rows
            for item in tree.get_children():
                row_num = tree.item(item, 'text')
                values = tree.item(item, 'values')
                data += '\t'.join([row_num] + [str(v) for v in values]) + '\n'
            
            self.clipboard_clear()
            self.clipboard_append(data)
            messagebox.showinfo("Success", f"Copied {len(tree.get_children())} rows to clipboard!")
        except Exception as e:
            logger.exception(f"Copy error: {e}")
            messagebox.showerror("Error", f"Failed to copy:\n{str(e)}")
    
    def _export_table_data(self, data_list, columns):
        """Export table data to CSV."""
        try:
            from tkinter import filedialog
            import csv
            
            filename = filedialog.asksaveasfilename(
                title="Export data",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(data_list)
                
                messagebox.showinfo("Success", f"Exported {len(data_list)} rows to:\n{filename}")
        except Exception as e:
            logger.exception(f"Export error: {e}")
            messagebox.showerror("Error", f"Failed to export:\n{str(e)}")
    
    def _show_column_stats(self, data_list, columns):
        """Show statistical information about columns."""
        try:
            from tkinter import scrolledtext
            
            stats_window = tk.Toplevel(self)
            stats_window.title("📊 Column Statistics")
            stats_window.geometry("700x600")
            
            # Header
            header = ttk.Frame(stats_window)
            header.pack(fill='x', padx=10, pady=10)
            ttk.Label(header, text="📊 Data Column Statistics", font=('', 12, 'bold')).pack(side='left')
            ttk.Label(header, text=f"{len(data_list)} rows  •  {len(columns)} columns", 
                     font=('', 10)).pack(side='right')
            
            # Stats display with tabs
            notebook = ttk.Notebook(stats_window)
            notebook.pack(fill='both', expand=True, padx=10, pady=5)
            
            # Tab 1: Overview
            overview_frame = ttk.Frame(notebook)
            notebook.add(overview_frame, text="Overview")
            
            overview_text = scrolledtext.ScrolledText(overview_frame, wrap=tk.WORD, font=('Courier New', 9))
            overview_text.pack(fill='both', expand=True, padx=5, pady=5)
            
            overview_text.insert('end', f"{'='*70}\n")
            overview_text.insert('end', f"DATASET OVERVIEW\n")
            overview_text.insert('end', f"{'='*70}\n\n")
            overview_text.insert('end', f"Total Rows: {len(data_list):,}\n")
            overview_text.insert('end', f"Total Columns: {len(columns)}\n\n")
            
            # Tab 2: Column Details
            details_frame = ttk.Frame(notebook)
            notebook.add(details_frame, text="Column Details")
            
            details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, font=('Courier New', 9))
            details_text.pack(fill='both', expand=True, padx=5, pady=5)
            
            for col in columns:
                details_text.insert('end', f"\n{'='*70}\n")
                details_text.insert('end', f"Column: {col}\n")
                details_text.insert('end', f"{'='*70}\n")
                
                # Collect values
                values = [row.get(col, '') for row in data_list]
                non_empty = [v for v in values if v != '' and v is not None]
                
                details_text.insert('end', f"Total values: {len(values)}\n")
                details_text.insert('end', f"Non-empty: {len(non_empty)}\n")
                details_text.insert('end', f"Empty: {len(values) - len(non_empty)}\n")
                details_text.insert('end', f"Fill rate: {len(non_empty)/len(values)*100:.1f}%\n")
                
                if non_empty:
                    # Check if numeric
                    numeric_values = []
                    for v in non_empty:
                        try:
                            numeric_values.append(float(v))
                        except (ValueError, TypeError):
                            pass
                    
                    if numeric_values and len(numeric_values) > len(non_empty) * 0.5:
                        # Mostly numeric
                        details_text.insert('end', f"Type: Numeric\n")
                        details_text.insert('end', f"Min: {min(numeric_values):.2f}\n")
                        details_text.insert('end', f"Max: {max(numeric_values):.2f}\n")
                        details_text.insert('end', f"Mean: {sum(numeric_values)/len(numeric_values):.2f}\n")
                    else:
                        # Text
                        details_text.insert('end', f"Type: Text\n")
                        unique_values = set(str(v) for v in non_empty)
                        details_text.insert('end', f"Unique values: {len(unique_values)}\n")
                        
                        # Show sample values
                        if len(unique_values) <= 10:
                            details_text.insert('end', f"Values: {', '.join(sorted(unique_values))}\n")
                        else:
                            sample = list(sorted(unique_values))[:5]
                            details_text.insert('end', f"Sample: {', '.join(sample)}...\n")
                
                details_text.insert('end', "\n")
            
            overview_text.config(state='disabled')
            details_text.config(state='disabled')
            
            # Close button
            ttk.Button(stats_window, text="Close", command=stats_window.destroy, 
                      width=15).pack(pady=10, ipady=5)
            
        except Exception as e:
            logger.exception(f"Stats error: {e}")
            messagebox.showerror("Error", f"Failed to generate statistics:\n{str(e)}")

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
            
            # Initialize automation instance
            self.automation_instance = self.automation_class(self.agent, config)
            
            # Verify instance was created
            if self.automation_instance is None:
                logger.error(f"Failed to create {self.tab_name} automation instance")
                return False
            
            # Run automation
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
                # Create result directory if it doesn't exist
                result_dir = DEFAULT_PATHS['results']
                os.makedirs(result_dir, exist_ok=True)
                
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(result_dir, filename)
                import cv2
                cv2.imwrite(filepath, screenshot)
                messagebox.showinfo("Success", f"Screenshot saved:\n{filepath}")
                logger.info(f"Screenshot saved: {filepath}")
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
