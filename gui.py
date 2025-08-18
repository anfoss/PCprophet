import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import io
import sys
import os


class PCprophetGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PCprophet")
        self.geometry("1200x800")

        self.stdout = io.StringIO()
        sys.stdout = self.stdout

        self.files_table_data = []

        # fixed choices
        self.group_options = ["1", "2", "3", "4"]
        self.cond_options = ["Ctrl", "Treat1", "Treat2", "Treat3", "Treat4"]

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
        ttk.Button(btn_frame, text="Export TSV", command=self.export_tsv).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Load Experimental Design", command=self.load_design_file).pack(side='left', padx=5)  # 👈 new button


        # Treeview for files and experiment mapping
        columns = ["Sample", "cond", "group", "short_id", "repl", "fr"]
        self.tree = ttk.Treeview(top_frame, columns=columns, show='headings', selectmode='extended')
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        self.tree.pack(fill='both', expand=True)

        # Bind double-click to edit cell
        self.tree.bind('<Double-1>', self.on_double_click)

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
            filetypes=[
                ("TSV files", "*.tsv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )
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

        # Dropdown for cond and group
        if col_index == 1:  # cond
            widget = ttk.Combobox(self.tree, values=self.cond_options, state="readonly")
        elif col_index == 2:  # group
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
        save_path = filedialog.asksaveasfilename(
            defaultextension=".tsv",
            filetypes=[("TSV files", "*.tsv")]
        )
        if not save_path:
            return

        rows = []
        for item in self.tree.get_children():
            rows.append(self.tree.item(item)['values'])
        df = pd.DataFrame(rows, columns=["Sample", "cond", "group", "short_id", "repl", "fr"])
        df.to_csv(save_path, sep="\t", index=False)
        self.update_stdout(f"Exported to {save_path}")


    def load_design_file(self):
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = filedialog.askopenfilename(
            title="Select experimental design TSV",
            filetypes=[("TSV files", "*.tsv"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=SCRIPT_DIR
        )

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

        # clear existing
        self.clear_files()

        # load rows
        for _, row in df.iterrows():
            values = [row[c] for c in required_cols]
            self.tree.insert("", "end", values=values)

        self.update_stdout(f"Loaded design file: {file_path}")

    # --- Run ---
    def run_pipeline(self):
        print("Run clicked")
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            print(values)
        self.update_stdout("Pipeline run finished.")

    # --- Logging ---
    def update_stdout(self, msg=None):
        if msg:
            print(msg)
            sys.stdout.flush()
        self.stdout_text.delete(1.0, tk.END)
        self.stdout_text.insert(tk.END, self.stdout.getvalue())


if __name__ == "__main__":
    app = PCprophetGUI()
    app.mainloop()
