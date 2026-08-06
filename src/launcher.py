"""
launcher.py

A tiny Tkinter picker window: choose a tab file (from data/, or Browse
for any other .txt) and type in the song's BPM once, with actual
labeled widgets instead of a terminal prompt. Only appears when
`main.py` is run with NO tab file argument at all -- passing one on
the command line (as before) skips this entirely, so scripted/
automated runs are unaffected.

Once "Start" is clicked, this window closes and `main.py` opens the
REAL visualizer in its own separate window, per the user's request --
two windows, one small one-time picker and one for the actual
animation, not one trying to be both.
"""

import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox


def prompt_for_inputs(data_dir: str = "data") -> tuple[str, float | None] | None:
    """
    Blocks (own Tk mainloop) until the user clicks Start or closes the
    window. Returns (tab_file_path, bpm_or_None) on Start, or None if
    the window was closed/cancelled instead.
    """
    result: dict = {"value": None}

    root = tk.Tk()
    root.title("Guitar Tab Visualizer")
    root.resizable(False, False)

    pad = {"padx": 12, "pady": 6}

    tk.Label(root, text="1. Choose a tab file:", font=("Segoe UI", 10, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", **pad
    )

    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    listbox = tk.Listbox(root, height=min(8, max(3, len(files))), width=44, exportselection=False)
    for f in files:
        listbox.insert(tk.END, os.path.basename(f))
    if files:
        listbox.selection_set(0)
    listbox.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12)

    browsed_path = {"path": None}

    def on_browse():
        path = filedialog.askopenfilename(
            title="Choose a tab file", filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            browsed_path["path"] = path
            listbox.selection_clear(0, tk.END)
            browse_label.config(text=f"Using: {os.path.basename(path)}")

    tk.Button(root, text="Browse for another file...", command=on_browse).grid(
        row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 0)
    )
    browse_label = tk.Label(root, text="", fg="#4C6EAF")
    browse_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=12)

    tk.Label(root, text="2. Song's BPM (optional):", font=("Segoe UI", 10, "bold")).grid(
        row=4, column=0, columnspan=2, sticky="w", **pad
    )
    tk.Label(
        root,
        text="Leave blank to fall back to rough default timing --\n"
        "BPM meaningfully improves playback accuracy if you know it.",
        fg="#666666", justify="left",
    ).grid(row=5, column=0, columnspan=2, sticky="w", padx=12)

    bpm_var = tk.StringVar()
    tk.Entry(root, textvariable=bpm_var, width=12).grid(row=6, column=0, sticky="w", padx=12, pady=(2, 12))

    def on_start():
        if browsed_path["path"]:
            tab_path = browsed_path["path"]
        elif listbox.curselection():
            tab_path = files[listbox.curselection()[0]]
        else:
            messagebox.showwarning("No file chosen", "Pick a tab file from the list, or Browse for one.")
            return

        bpm_text = bpm_var.get().strip()
        bpm = None
        if bpm_text:
            try:
                bpm = float(bpm_text)
            except ValueError:
                messagebox.showwarning("Invalid BPM", f"'{bpm_text}' isn't a number -- leave it blank or fix it.")
                return

        result["value"] = (tab_path, bpm)
        root.destroy()

    button_row = tk.Frame(root)
    button_row.grid(row=7, column=0, columnspan=2, sticky="e", padx=12, pady=(0, 12))
    tk.Button(button_row, text="Cancel", command=root.destroy).pack(side="right", padx=(6, 0))
    tk.Button(button_row, text="Start ▶", command=on_start, default="active").pack(side="right")

    root.bind("<Return>", lambda _event: on_start())
    root.mainloop()

    return result["value"]
