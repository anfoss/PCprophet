import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import io
import sys

class PCprophetGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PCprophet")
        self.geometry("1000x700")

        self.stdout = io.StringIO()
        sys.stdout = self.stdout

        self.df = None
        self.dropdown_vars = {}

        self.create_widgets()

    def create_widgets(self):
        button_frame = ttk.Frame(self)
        button_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky='ew')

        self.select_sample_ids_button = ttk.Button(button_frame, text="Select Sample IDs", command=self.select_sample_ids)
        self.select_sample_ids_button.pack(side='left', padx=5)

        self.select_output_folder_button = ttk.Button(button_frame, text="Select Output Folder", command=self.select_output_folder)
        self.select_output_folder_button.pack(side='left', padx=5)

        self.select_config_button = ttk.Button(button_frame, text="Select Config Settings", command=self.select_config)
        self.select_config_button.pack(side='left', padx=5)

        self.load_metadata_button = ttk.Button(button_frame, text="Load Metadata CSV", command=self.load_metadata_csv)
        self.load_metadata_button.pack(side='left', padx=5)

        options_frame = ttk.Frame(self)
        options_frame.grid(row=1, column=0, pady=10, sticky='nw')

        self.skip_feature_gen_var = tk.BooleanVar()
        self.skip_feature_gen_check = ttk.Checkbutton(options_frame, text="Skip Feature Generation", variable=self.skip_feature_gen_var)
        self.skip_feature_gen_check.grid(row=0, column=0, sticky='w', pady=5)

        self.ppi_fdr_label = ttk.Label(options_frame, text="Gene ontology FDR (0-1):")
        self.ppi_fdr_label.grid(row=1, column=0, sticky='w')

        self.ppi_fdr_var = tk.DoubleVar()
        self.ppi_fdr_entry = ttk.Entry(options_frame, textvariable=self.ppi_fdr_var)
        self.ppi_fdr_entry.grid(row=2, column=0, sticky='w', pady=5)

        self.run_button = ttk.Button(options_frame, text="Run", command=self.run)
        self.run_button.grid(row=3, column=0, sticky='w', pady=5)

        self.stdout_text = tk.Text(self, wrap='word', height=20)
        self.stdout_text.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky='nsew')

        self.metadata_table_frame = ttk.Frame(self)
        self.metadata_table_frame.grid(row=2, column=0, sticky='nw')

        self.mapping_frame = ttk.Frame(self)
        self.mapping_frame.grid(row=3, column=0, sticky='nw')

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def select_sample_ids(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            print(f"Selected sample IDs file: {file_path}")
            self.update_stdout()

    def select_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            print(f"Selected output folder: {folder_path}")
            self.update_stdout()

    def select_config(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            print(f"Selected config settings file: {file_path}")
            self.update_stdout()

    def load_metadata_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            print(f"Loaded metadata CSV: {file_path}")
            self.update_stdout()
            self.display_metadata_table()
            self.setup_column_mapping()

    def display_metadata_table(self):
        for widget in self.metadata_table_frame.winfo_children():
            widget.destroy()

        if self.df is not None:
            tree = ttk.Treeview(self.metadata_table_frame, columns=self.df.columns.tolist(), show='headings')
            for col in self.df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            for _, row in self.df.iterrows():
                tree.insert("", "end", values=list(row))
            tree.pack(fill="both", expand=True)

    def setup_column_mapping(self):
        for widget in self.mapping_frame.winfo_children():
            widget.destroy()

        required_fields = ['Sample', 'cond', 'group', 'short_id', 'repl', 'fr']
        ttk.Label(self.mapping_frame, text="Map columns:").pack()

        for field in required_fields:
            frame = ttk.Frame(self.mapping_frame)
            frame.pack(pady=2, anchor="w")
            label = ttk.Label(frame, text=f"{field} →")
            label.pack(side="left")
            var = tk.StringVar()
            dropdown = ttk.Combobox(frame, textvariable=var, values=self.df.columns.tolist(), state='readonly')
            dropdown.pack(side="left")
            self.dropdown_vars[field] = var

        confirm_btn = ttk.Button(self.mapping_frame, text="Confirm Mapping", command=self.confirm_mapping)
        confirm_btn.pack(pady=5)

    def confirm_mapping(self):
        mapping = {field: var.get() for field, var in self.dropdown_vars.items()}
        print("Confirmed Mapping:", mapping)
        self.update_stdout()

    def run(self):
        print("Run button clicked")
        self.update_stdout()

    def update_stdout(self):
        self.stdout_text.delete(1.0, tk.END)
        self.stdout_text.insert(tk.END, self.stdout.getvalue())

if __name__ == "__main__":
    app = PCprophetGUI()
    app.mainloop()
