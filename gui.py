import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import io
import sys
import os
import configparser
import main 
import subprocess


class SettingsPanel(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Settings")

        self.entries = {}

        # Parameter definitions (label -> default, widget type, choices if dropdown)
        self.params = {
            "db": ("meta/corum_allComplexes.txt", "file"),
            "sid": ("sample_ids.txt", "file"),
            "output": ("./Output", "folder"),
            "cal": ("None", "file"),
            "mw": ("None", "file"),
            "is_ppi": ("False", "dropdown", ["True", "False"]),
            "all_fract": ("all", "entry"),
            "merge": ("all", "dropdown", ["all", "reference"]),
            "fdr": (0.7, "entry"),
            "collapse_mode": ("GO", "dropdown", ["GO", "CAL", "SUPER", "PROB", "NONE"]),
            "skip": ("False", "dropdown", ["True", "False"]),
            "diff": ("True", "dropdown", ["True", "False"]),
        }

        for i, (key, val) in enumerate(self.params.items()):
            default, wtype, *extra = val
            ttk.Label(self, text=key).grid(row=i, column=0, sticky="w", padx=5, pady=2)

            if wtype == "file":
                frame = ttk.Frame(self)
                entry = ttk.Entry(frame, width=40)
                entry.insert(0, default)
                entry.pack(side="left", fill="x", expand=True)
                ttk.Button(frame, text="Browse", command=lambda e=entry: self.browse_file(e)).pack(side="right")
                frame.grid(row=i, column=1, sticky="ew", padx=5)
                self.entries[key] = entry

            elif wtype == "folder":
                frame = ttk.Frame(self)
                entry = ttk.Entry(frame, width=40)
                entry.insert(0, default)
                entry.pack(side="left", fill="x", expand=True)
                ttk.Button(frame, text="Browse", command=lambda e=entry: self.browse_folder(e)).pack(side="right")
                frame.grid(row=i, column=1, sticky="ew", padx=5)
                self.entries[key] = entry

            elif wtype == "dropdown":
                values = extra[0]
                combo = ttk.Combobox(self, values=values, state="readonly")
                combo.set(default)
                combo.grid(row=i, column=1, sticky="ew", padx=5)
                self.entries[key] = combo

            else:  # entry
                entry = ttk.Entry(self)
                entry.insert(0, default)
                entry.grid(row=i, column=1, sticky="ew", padx=5)
                self.entries[key] = entry

        # Config load/save buttons
        btn_frame = ttk.Frame(self)
        ttk.Button(btn_frame, text="Load Config", command=self.load_config).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Save Config", command=self.save_config).pack(side="left", padx=5)
        btn_frame.grid(row=len(self.params), column=0, columnspan=2, pady=10)

        self.columnconfigure(1, weight=1)

    def browse_file(self, entry):
        path = filedialog.askopenfilename()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def browse_folder(self, entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def load_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("Config files", "*.conf")])
        if not file_path:
            return
        config = configparser.ConfigParser()
        config.read(file_path)
        for section in config.sections():
            for key, val in config[section].items():
                if key in self.entries:
                    self.entries[key].delete(0, tk.END)
                    self.entries[key].insert(0, val)

    def save_temp_config(self):
        """
        Write current settings to a temporary config file.
        Returns the path to that file.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".conf")
        config = configparser.ConfigParser()
        config["GLOBAL"] = {k: self.entries[k].get() for k in ["db", "sid", "output", "cal", "mw", "skip", "diff"]}
        config["PREPROCESS"] = {k: self.entries[k].get() for k in ["is_ppi", "all_fract", "merge"]}
        config["POSTPROCESS"] = {k: self.entries[k].get() for k in ["fdr", "collapse_mode"]}
        config.write(temp_file)
        temp_file.close()
        return temp_file.name


    def save_config(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".conf", filetypes=[("Config files", "*.conf")])
        if not file_path:
            return
        config = configparser.ConfigParser()
        config["GLOBAL"] = {k: self.entries[k].get() for k in ["db", "sid", "output", "cal", "mw", "skip", "diff"]}
        config["PREPROCESS"] = {k: self.entries[k].get() for k in ["is_ppi", "all_fract", "merge"]}
        config["POSTPROCESS"] = {k: self.entries[k].get() for k in ["fdr", "collapse_mode"]}
        with open(file_path, "w") as f:
            config.write(f)

    def get_config_dict(self):
        """Return config dict in same format pipeline.run_pipeline expects"""
        return {
            "GLOBAL": {
                "db": self.entries["db"].get(),
                "sid": self.entries["sid"].get(),
                "output": self.entries["output"].get(),
                "cal": self.entries["cal"].get(),
                "mw": self.entries["mw"].get(),
                "temp": "./tmp",
                "skip": self.entries["skip"].get(),
                "diff": self.entries["diff"].get(),
                "go_obo": "meta/go_graph.graphml",
                "sp_go": "meta/go_gaf.pkl",
                "rf": "meta/rf_allneg.pkl",
            },
            "PREPROCESS": {
                "is_ppi": self.entries["is_ppi"].get(),
                "all_fract": self.entries["all_fract"].get(),
                "merge": self.entries["merge"].get(),
            },
            "POSTPROCESS": {
                "fdr": float(self.entries["fdr"].get()),
                "collapse_mode": self.entries["collapse_mode"].get(),
            }
        }


class PCprophetGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PCprophet")
        self.geometry("1200x800")

        self.stdout = io.StringIO()
        sys.stdout = self.stdout
        self.files_table_data = []

        self.group_options = ["1", "2", "3", "4"]
        self.cond_options = ["Ctrl", "Treat1", "Treat2", "Treat3", "Treat4"]

        # --- Notebook Tabs ---
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        # Experimental design tab
        self.design_tab = ttk.Frame(notebook)
        notebook.add(self.design_tab, text="Experimental Design")
        self.create_design_panel(self.design_tab)

        # Settings tab
        self.settings_tab = ttk.Frame(notebook)
        notebook.add(self.settings_tab, text="Settings")
        self.settings_panel = SettingsPanel(self.settings_tab)
        self.settings_panel.pack(fill="both", expand=True, padx=10, pady=10)

        # Logs tab
        self.logs_tab = ttk.Frame(notebook)
        notebook.add(self.logs_tab, text="Run / Logs")
        self.create_log_panel(self.logs_tab)

    def create_design_panel(self, parent):
        top_frame = ttk.LabelFrame(parent, text="Experimental design")
        top_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Buttons above table
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Load Files", command=self.load_files).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Clear Selection", command=self.clear_files).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Export TSV", command=self.export_tsv).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Load Experimental Design", command=self.load_design_file).pack(side='left', padx=5)

        columns = ["Sample", "cond", "group", "short_id", "repl", "fr"]
        self.tree = ttk.Treeview(top_frame, columns=columns, show='headings', selectmode='extended')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        self.tree.pack(fill='both', expand=True)

        self.tree.bind('<Double-1>', self.on_double_click)

    def create_log_panel(self, parent):
        bottom_frame = ttk.LabelFrame(parent, text="Run / Logs")
        bottom_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.run_button = ttk.Button(bottom_frame, text="Run", command=self.run_pipeline)
        self.run_button.pack(anchor='w', pady=5, padx=5)

        self.stdout_text = tk.Text(bottom_frame, wrap='word', height=20)
        self.stdout_text.pack(fill='both', expand=True, padx=5, pady=5)

    # --- File functions ---
    def load_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("TSV files", "*.tsv"), ("Text files", "*.txt"), ("All files", "*.*")])
        for path in paths:
            row = [path, "", "", "", "", ""]
            self.files_table_data.append(row)
            self.tree.insert("", "end", values=row)
        self.update_stdout(f"Loaded {len(paths)} files.")

    def clear_files(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.files_table_data.clear()
        self.update_stdout("Cleared all file selections.")

    # --- Double click editing ---
    def on_double_click(self, event):
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not item or not column:
            return
        col_index = int(column.replace("#", "")) - 1
        old_value = self.tree.item(item, "values")[col_index]
        x, y, width, height = self.tree.bbox(item, column)

        if col_index == 1:
            widget = ttk.Combobox(self.tree, values=self.cond_options, state="readonly")
        elif col_index == 2:
            widget = ttk.Combobox(self.tree, values=self.group_options, state="readonly")
        else:
            widget = tk.Entry(self.tree)

        widget.place(x=x, y=y, width=width, height=height)
        widget.insert(0, old_value)
        widget.focus()

        def save_edit(event=None):
            new_value = widget.get()
            values = list(self.tree.item(item, "values"))
            values[col_index] = new_value
            self.tree.item(item, values=values)
            widget.destroy()

        if isinstance(widget, ttk.Combobox):
            widget.bind("<<ComboboxSelected>>", save_edit)
            widget.bind("<FocusOut>", lambda e: widget.destroy())
        else:
            widget.bind("<Return>", save_edit)
            widget.bind("<FocusOut>", lambda e: widget.destroy())

    # --- Export table ---
    def export_tsv(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".tsv", filetypes=[("TSV files", "*.tsv")])
        if not save_path:
            return
        rows = [self.tree.item(item)['values'] for item in self.tree.get_children()]
        df = pd.DataFrame(rows, columns=["Sample", "cond", "group", "short_id", "repl", "fr"])
        df.to_csv(save_path, sep="\t", index=False)
        self.update_stdout(f"Exported to {save_path}")

    def load_design_file(self):
        file_path = filedialog.askopenfilename(title="Select experimental design TSV", filetypes=[("TSV files", "*.tsv"), ("Text files", "*.txt"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            self.update_stdout(f"Error loading design file: {e}")
            return
        required_cols = ["Sample", "cond", "group", "short_id", "repl", "fr"]
        if not all(col in df.columns for col in required_cols):
            self.update_stdout("Error: Design file missing required columns")
            return
        self.clear_files()
        for _, row in df.iterrows():
            values = [row[c] for c in required_cols]
            self.tree.insert("", "end", values=values)
        self.update_stdout(f"Loaded design file: {file_path}")


    def run_pipeline(self):
        config_dict = self.settings_panel.get_config_dict()
        args = config_to_args(config_dict)
        cmd = ["python3", "main.py"] + args
        print(f"Running command: {' '.join(cmd)}")
        proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

        def poll_output():
            line = proc.stdout.readline()
            if line:
                text_widget.insert(tk.END, line)
                text_widget.see(tk.END)
                text_widget.update_idletasks()
            if proc.poll() is None:
                text_widget.after(100, poll_output)

        poll_output()


    # --- Logs ---
    def update_stdout(self, msg=None):
        if msg:
            print(msg)
            sys.stdout.flush()
        self.stdout_text.delete(1.0, tk.END)
        self.stdout_text.insert(tk.END, self.stdout.getvalue())


def config_to_args(config_dict):
    """
    Convert nested config dict to a flat list of CLI args.
    Example: {"GLOBAL": {"sid": "sample.txt", "db": "db.txt"}}
    → ["-sid", "sample.txt", "-db", "db.txt"]
    """
    args = []
    for section, options in config_dict.items():
        for key, value in options.items():
            if value is None or value == "":
                continue
            args.append(f"-{key}")
            args.append(str(value))
    return args



if __name__ == "__main__":
    app = PCprophetGUI()
    app.mainloop()
