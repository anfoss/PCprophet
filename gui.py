import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import io
import sys

import tkinter as tk
from tkinter import ttk, filedialog
import io
import sys

class PCprophetGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PCprophet")
        self.geometry("1200x800")

        self.stdout = io.StringIO()
        sys.stdout = self.stdout

        self.files_table_data = []

        self.create_widgets()

    def create_widgets(self):
        # --- Top Panel: File Organizer ---
        top_frame = ttk.LabelFrame(self, text="Experimental design")
        top_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=5)

        # Buttons above the table
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Load Files", command=self.load_files).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Clear Selection", command=self.clear_files).pack(side='left', padx=5)

        # Treeview for files and experiment mapping
        columns = ["Path", "Condition", "Bioreplicate", "Short Name", "Group"]
        self.tree = ttk.Treeview(top_frame, columns=columns, show='headings', selectmode='extended')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=200)
        self.tree.pack(fill='both', expand=True)

        # --- Bottom Panel: Logs and Run ---
        bottom_frame = ttk.LabelFrame(self, text="Run / Logs")
        bottom_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)

        self.run_button = ttk.Button(bottom_frame, text="Run", command=self.run_pipeline)
        self.run_button.pack(anchor='w', pady=5, padx=5)

        self.stdout_text = tk.Text(bottom_frame, wrap='word', height=20)
        self.stdout_text.pack(fill='both', expand=True, padx=5, pady=5)

        self.grid_rowconfigure(0, weight=3)
        self.grid_rowconfigure(1, weight=2)
        self.grid_columnconfigure(0, weight=1)

    # --- File loading functions ---
    def load_files(self):
        paths = filedialog.askopenfilenames(
            title="Select files",
            filetypes=[("TSV and TXT files", "*.tsv *.txt")],
            multiple=True
        )
        for path in paths:
            self.files_table_data.append([path, "", "", "", ""])
            self.tree.insert("", "end", values=[path, "", "", "", ""])
        self.update_stdout(f"Loaded {len(paths)} files.")

    def clear_files(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.files_table_data.clear()
        self.update_stdout("Cleared all file selections.")

    # --- Run ---
    def run_pipeline(self):
        print("Run clicked")
        # Here you could loop over self.tree to get user-defined experiment/replicate info
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            print(values)
        self.update_stdout("Pipeline run finished.")

    # --- Logging ---
    def update_stdout(self, msg=None):
        if msg:
            print(msg)
        self.stdout_text.delete(1.0, tk.END)
        self.stdout_text.insert(tk.END, self.stdout.getvalue())


if __name__ == "__main__":
    app = PCprophetGUI()
    app.mainloop()


# class PCprophetGUI(tk.Tk):
#     def __init__(self):
#         super().__init__()

#         self.title("PCprophet")
#         self.geometry("1100x800")

#         self.stdout = io.StringIO()
#         sys.stdout = self.stdout

#         self.df = None
#         self.dropdown_vars = {}
#         self.gui_vars = {}

#         self.create_widgets()

#     def create_widgets(self):
#         # --- File Selection Frame ---
#         file_frame = ttk.LabelFrame(self, text="File Selection")
#         file_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky='ew')

#         ttk.Button(file_frame, text="Select Database", command=self.select_database).grid(row=0, column=0, padx=5, pady=5)
#         ttk.Button(file_frame, text="Select Sample IDs", command=self.select_sample_ids).grid(row=0, column=1, padx=5, pady=5)
#         ttk.Button(file_frame, text="Select Output Folder", command=self.select_output_folder).grid(row=0, column=2, padx=5, pady=5)
#         ttk.Button(file_frame, text="Select Calibration File", command=self.select_calibration).grid(row=1, column=0, padx=5, pady=5)
#         ttk.Button(file_frame, text="Select MW UniProt File", command=self.select_mwuni).grid(row=1, column=1, padx=5, pady=5)

#         # --- Options Frame ---
#         options_frame = ttk.LabelFrame(self, text="Options")
#         options_frame.grid(row=1, column=0, sticky='nw', padx=10, pady=10)

#         # skip feature generation
#         self.gui_vars['skip'] = tk.BooleanVar(value=False)
#         ttk.Checkbutton(options_frame, text="Skip Feature Generation", variable=self.gui_vars['skip']).grid(row=0, column=0, sticky='w', pady=5)

#         # is_ppi dropdown
#         ttk.Label(options_frame, text="Is PPI Database?").grid(row=1, column=0, sticky='w')
#         self.gui_vars['is_ppi'] = tk.StringVar(value="False")
#         ttk.Combobox(options_frame, textvariable=self.gui_vars['is_ppi'], values=["True", "False"], state='readonly').grid(row=2, column=0, sticky='w', pady=5)

#         # all fractions dropdown
#         ttk.Label(options_frame, text="Use Fractions").grid(row=3, column=0, sticky='w')
#         self.gui_vars['all_fract'] = tk.StringVar(value='all')
#         ttk.Combobox(options_frame, textvariable=self.gui_vars['all_fract'], values=['all', '1-X'], state='readonly').grid(row=4, column=0, sticky='w', pady=5)

#         # merge mode
#         ttk.Label(options_frame, text="Merge Mode").grid(row=5, column=0, sticky='w')
#         self.gui_vars['merge'] = tk.StringVar(value='all')
#         ttk.Combobox(options_frame, textvariable=self.gui_vars['merge'], values=['all', 'reference'], state='readonly').grid(row=6, column=0, sticky='w', pady=5)

#         # collapse mode
#         ttk.Label(options_frame, text="Collapse Mode").grid(row=7, column=0, sticky='w')
#         self.gui_vars['collapse'] = tk.StringVar(value='GO')
#         ttk.Combobox(options_frame, textvariable=self.gui_vars['collapse'], values=['GO', 'CAL', 'SUPER', 'PROB', 'NONE'], state='readonly').grid(row=8, column=0, sticky='w', pady=5)

#         # FDR slider
#         ttk.Label(options_frame, text="FDR (0-1)").grid(row=9, column=0, sticky='w')
#         self.gui_vars['fdr'] = tk.DoubleVar(value=0.2)
#         self.fdr_entry = ttk.Entry(options_frame, textvariable=self.gui_vars['fdr'], width=10)
#         self.fdr_entry.grid(row=10, column=0, sticky='w', pady=5)

#         # differential analysis checkbox
#         self.gui_vars['dif'] = tk.BooleanVar(value=True)
#         ttk.Checkbutton(options_frame, text="Perform Differential Analysis", variable=self.gui_vars['dif']).grid(row=11, column=0, sticky='w', pady=5)

#         # --- Metadata / Column Mapping ---
#         self.load_metadata_button = ttk.Button(options_frame, text="Load Metadata CSV", command=self.load_metadata_csv)
#         self.load_metadata_button.grid(row=12, column=0, sticky='w', pady=5)

#         # --- Run Button ---
#         self.run_button = ttk.Button(options_frame, text="Run", command=self.run)
#         self.run_button.grid(row=13, column=0, sticky='w', pady=10)

#         # --- Output / Logs ---
#         self.stdout_text = tk.Text(self, wrap='word', height=25)
#         self.stdout_text.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky='nsew')

#         self.metadata_table_frame = ttk.Frame(self)
#         self.metadata_table_frame.grid(row=2, column=0, sticky='nw')

#         self.mapping_frame = ttk.Frame(self)
#         self.mapping_frame.grid(row=3, column=0, sticky='nw')

#         self.grid_columnconfigure(1, weight=1)
#         self.grid_rowconfigure(1, weight=1)

#     # --- File selectors ---
#     def select_database(self):
#         path = filedialog.askopenfilename()
#         if path:
#             self.gui_vars['database'] = path
#             print(f"Selected database: {path}")
#             self.update_stdout()

#     def select_sample_ids(self):
#         path = filedialog.askopenfilename()
#         if path:
#             self.gui_vars['sample_ids'] = path
#             print(f"Selected sample IDs: {path}")
#             self.update_stdout()

#     def select_output_folder(self):
#         path = filedialog.askdirectory()
#         if path:
#             self.gui_vars['out_folder'] = path
#             print(f"Selected output folder: {path}")
#             self.update_stdout()

#     def select_calibration(self):
#         path = filedialog.askopenfilename()
#         if path:
#             self.gui_vars['calibration'] = path
#             print(f"Selected calibration file: {path}")
#             self.update_stdout()

#     def select_mwuni(self):
#         path = filedialog.askopenfilename()
#         if path:
#             self.gui_vars['mwuni'] = path
#             print(f"Selected MW UniProt file: {path}")
#             self.update_stdout()

#     # --- Metadata / Mapping ---
#     def load_metadata_csv(self):
#         path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
#         if path:
#             self.df = pd.read_csv(path)
#             print(f"Loaded metadata CSV: {path}")
#             self.update_stdout()
#             self.display_metadata_table()
#             self.setup_column_mapping()

#     def display_metadata_table(self):
#         for widget in self.metadata_table_frame.winfo_children():
#             widget.destroy()

#         if self.df is not None:
#             tree = ttk.Treeview(self.metadata_table_frame, columns=self.df.columns.tolist(), show='headings')
#             for col in self.df.columns:
#                 tree.heading(col, text=col)
#                 tree.column(col, width=100)
#             for _, row in self.df.iterrows():
#                 tree.insert("", "end", values=list(row))
#             tree.pack(fill="both", expand=True)

#     def setup_column_mapping(self):
#         for widget in self.mapping_frame.winfo_children():
#             widget.destroy()

#         required_fields = ['Sample', 'cond', 'group', 'short_id', 'repl', 'fr']
#         ttk.Label(self.mapping_frame, text="Map columns:").pack()

#         for field in required_fields:
#             frame = ttk.Frame(self.mapping_frame)
#             frame.pack(pady=2, anchor="w")
#             label = ttk.Label(frame, text=f"{field} →")
#             label.pack(side="left")
#             var = tk.StringVar()
#             dropdown = ttk.Combobox(frame, textvariable=var, values=self.df.columns.tolist(), state='readonly')
#             dropdown.pack(side="left")
#             self.dropdown_vars[field] = var

#         confirm_btn = ttk.Button(self.mapping_frame, text="Confirm Mapping", command=self.confirm_mapping)
#         confirm_btn.pack(pady=5)

#     def confirm_mapping(self):
#         mapping = {field: var.get() for field, var in self.dropdown_vars.items()}
#         print("Confirmed Mapping:", mapping)
#         self.update_stdout()

#     # --- Run ---
#     def run(self):
#         print("Run clicked with settings:")
#         for key, var in self.gui_vars.items():
#             value = var.get() if isinstance(var, (tk.StringVar, tk.BooleanVar, tk.DoubleVar)) else var
#             print(f"{key}: {value}")
#         self.update_stdout()

#     def update_stdout(self):
#         self.stdout_text.delete(1.0, tk.END)
#         self.stdout_text.insert(tk.END, self.stdout.getvalue())


# if __name__ == "__main__":
#     app = PCprophetGUI()
#     app.mainloop()
