"""GUI launcher for the Memory Optimization Pipeline."""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading, queue, sys, os, io, time, json, subprocess, webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Memory Optimization Pipeline")
        self.root.geometry("1100x800")
        self.root.minsize(900, 600)
        self.process = None
        self.log_queue = queue.Queue()
        self.results_data = None
        self.output_dir = None

        self._build_ui()

    def _build_ui(self):
        # ── Top: controls ──
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Mode:").grid(row=0, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value="test")
        ttk.Radiobutton(ctrl, text="Test (5 built-in dialogues)", variable=self.mode_var,
                        value="test", command=self._on_mode_change).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(ctrl, text="User (single dialogue)", variable=self.mode_var,
                        value="user", command=self._on_mode_change).grid(row=0, column=2, sticky=tk.W, padx=5)

        ttk.Label(ctrl, text="Rounds (迭代轮次):").grid(row=0, column=3, sticky=tk.E, padx=(20, 0))
        self.rounds_var = tk.IntVar(value=5)
        ttk.Spinbox(ctrl, from_=1, to=10, textvariable=self.rounds_var, width=4).grid(row=0, column=4)

        ttk.Label(ctrl, text="K (每轮候选数):").grid(row=0, column=5, padx=(15, 0))
        self.cand_var = tk.IntVar(value=5)
        ttk.Spinbox(ctrl, from_=2, to=10, textvariable=self.cand_var, width=4).grid(row=0, column=6)

        # User input area (hidden in test mode)
        self.user_frame = ttk.Frame(ctrl)
        self.user_frame.grid(row=1, column=0, columnspan=7, sticky=tk.EW, pady=(10, 0))

        ttk.Label(self.user_frame, text="Dialogue text:").pack(anchor=tk.W)
        self.text_input = scrolledtext.ScrolledText(self.user_frame, height=5, wrap=tk.WORD)
        self.text_input.pack(fill=tk.X, pady=3)

        btn_row = ttk.Frame(self.user_frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Load from file...", command=self._pick_file).pack(side=tk.LEFT, padx=(0, 10))
        self.file_label = ttk.Label(btn_row, text="")
        self.file_label.pack(side=tk.LEFT)

        # Run button
        btn_frame = ttk.Frame(ctrl)
        btn_frame.grid(row=2, column=0, columnspan=7, pady=(10, 0))
        self.run_btn = ttk.Button(btn_frame, text="▶ Run", command=self._run, width=20)
        self.run_btn.pack()
        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate")
        self.status_label = ttk.Label(btn_frame, text="", foreground="gray")

        # ── Middle: log output ──
        mid = ttk.LabelFrame(self.root, text="Progress", padding=5)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_box = scrolledtext.ScrolledText(mid, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

        # ── Bottom: results ──
        bottom = ttk.LabelFrame(self.root, text="Results", padding=5)
        bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Notebook for tabs
        self.result_notebook = ttk.Notebook(bottom)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Best memory
        tab_mem = ttk.Frame(self.result_notebook)
        self.result_notebook.add(tab_mem, text="Best Memory")
        self.memory_text = scrolledtext.ScrolledText(tab_mem, wrap=tk.WORD, font=("Segoe UI", 11), state=tk.DISABLED)
        self.memory_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 2: Output files
        tab_files = ttk.Frame(self.result_notebook)
        self.result_notebook.add(tab_files, text="Output Files")
        self.files_listbox = tk.Listbox(tab_files, font=("Consolas", 10))
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.files_listbox.bind("<Double-Button-1>", self._open_selected_file)
        files_scroll = ttk.Scrollbar(tab_files, orient=tk.VERTICAL, command=self.files_listbox.yview)
        files_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.files_listbox.config(yscrollcommand=files_scroll.set)

        # Tab 3: Summary table
        tab_summary = ttk.Frame(self.result_notebook)
        self.result_notebook.add(tab_summary, text="Summary")
        self.summary_text = scrolledtext.ScrolledText(tab_summary, wrap=tk.NONE, font=("Consolas", 10), state=tk.DISABLED)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 4: Experience rules
        tab_rules = ttk.Frame(self.result_notebook)
        self.result_notebook.add(tab_rules, text="Rules")
        self.rules_text = scrolledtext.ScrolledText(tab_rules, wrap=tk.WORD, font=("Segoe UI", 10), state=tk.DISABLED)
        self.rules_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._on_mode_change()

    def _on_mode_change(self):
        if self.mode_var.get() == "user":
            self.user_frame.grid()
        else:
            self.user_frame.grid_remove()

    def _pick_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text/JSON", "*.txt;*.json"), ("All", "*.*")])
        if path:
            self.file_label.config(text=os.path.basename(path))
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                if path.endswith(".json"):
                    data = json.loads(text)
                    if isinstance(data, dict):
                        text = data.get("dialogue") or data.get("text", "")
                    elif isinstance(data, list) and data:
                        text = data[0].get("dialogue") or data[0].get("text", "")
                self.text_input.delete("1.0", tk.END)
                self.text_input.insert("1.0", text)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, f"[{ts}] {text}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def _run(self):
        self.run_btn.config(state=tk.DISABLED)
        self.progress.pack(fill=tk.X, pady=2)
        self.progress.start()
        self.status_label.config(text="RUNNING... (wait for DONE popup)", foreground="blue")
        self.status_label.pack()
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state=tk.DISABLED)
        self._clear_results()
        self._log(f"Starting pipeline: {self.mode_var.get()} mode, "
                  f"rounds={self.rounds_var.get()}, K={self.cand_var.get()}")

        threading.Thread(target=self._run_pipeline, daemon=True).start()
        self.root.after(100, self._poll_log)

    def _run_pipeline(self):
        try:
            mode = self.mode_var.get()
            rounds = self.rounds_var.get()
            cand = self.cand_var.get()

            # Use Python 3.12 (which has all deps), not Anaconda
            python_exe = r"C:\Users\z2cc_\AppData\Local\Programs\Python\Python312\python.exe"
            if not os.path.exists(python_exe):
                python_exe = sys.executable
            cmd = [python_exe, "main.py", "--rounds", str(rounds), "--candidates", str(cand)]

            if mode == "test":
                cmd += ["--mode", "test"]
                self.output_dir = "output/test"
            else:
                text = self.text_input.get("1.0", tk.END).strip()
                if not text:
                    self.log_queue.put("ERROR: No dialogue text provided")
                    return
                cmd += ["--mode", "user", "--text", text]
                self.output_dir = "output/user"

            self.log_queue.put(f"Running: {' '.join(cmd[:6])}...")
            self.log_queue.put("─" * 50)

            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            for line in self.process.stdout:
                line = line.rstrip()
                if line:
                    self.log_queue.put(line)

            self.process.wait()
            self.log_queue.put("─" * 50)
            self.log_queue.put(f"DONE (exit code {self.process.returncode})")
            self.log_queue.put("__DONE__")

        except Exception as e:
            self.log_queue.put(f"ERROR: {e}")
            self.log_queue.put("__DONE__")

    def _poll_log(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "__DONE__":
                    self.root.after(500, self._on_done)
                    return
                self._log(line)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _on_done(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.config(text="")
        self.run_btn.config(state=tk.NORMAL)

        if self.process and self.process.returncode != 0:
            self._log(f"ERROR: Pipeline exited with code {self.process.returncode}")
            self.status_label.config(text="FAILED", foreground="red")
            messagebox.showerror("Error", f"Pipeline failed (exit code {self.process.returncode}). Check the log above for details.")
            return

        if self.output_dir:
            results_path = os.path.join(self.output_dir, "results.json")
            if os.path.exists(results_path):
                mtime = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(results_path)))
                self._log(f"Results saved → {self.output_dir}/ (updated {mtime})")
                self.status_label.config(text="DONE", foreground="green")
                self._load_results()
                self.result_notebook.select(0)  # switch to Best Memory tab
                messagebox.showinfo("Done", f"Pipeline complete.\nResults → {os.path.abspath(self.output_dir)}/")
            else:
                self._log(f"WARNING: No results.json in {self.output_dir}")
                self.status_label.config(text="NO OUTPUT", foreground="orange")
                messagebox.showwarning("Warning", f"No results found.\nExpected: {os.path.abspath(results_path)}")

    def _clear_results(self):
        for widget in [self.memory_text, self.summary_text, self.rules_text]:
            widget.config(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            widget.config(state=tk.DISABLED)
        self.files_listbox.delete(0, tk.END)

    def _load_results(self):
        results_path = os.path.join(self.output_dir, "results.json")
        exp_path = os.path.join(self.output_dir, "results_experiences.json")

        # Load results
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                self.results_data = json.load(f)

            # Best memory tab — show ALL dialogues
            self.memory_text.config(state=tk.NORMAL)
            for r in self.results_data:
                did = r.get("dialogue_id", "?")
                mem = r.get("best_memory", "")
                score = r.get("best_score", 0)
                if isinstance(score, dict):
                    score = score.get("composite", 0)
                traj = r.get("score_trajectory", [])
                best = max(traj) if traj else score

                self.memory_text.insert(tk.END,
                    f"┌─ {did} ─────────────────────────────────────────────\n"
                    f"│ Score: {best:.4f}  |  Rules: {r.get('experience_count', 0)}\n"
                    f"└{'─'*60}\n{mem}\n\n\n")
            self.memory_text.config(state=tk.DISABLED)

            # Summary tab
            self.summary_text.config(state=tk.NORMAL)
            for r in self.results_data:
                traj = r.get("score_trajectory", [])
                did = r.get("dialogue_id", "?")
                r0 = traj[0] if traj else 0
                best = max(traj) if traj else 0
                d = best - r0
                rules = r.get("experience_count", 0)
                self.summary_text.insert(tk.END,
                    f"{did:<25} R0={r0:.4f}  Best={best:.4f}  Delta={d:+.4f}  Rules={rules}\n")
            self.summary_text.config(state=tk.DISABLED)

        # Experience rules tab
        if os.path.exists(exp_path):
            with open(exp_path, "r", encoding="utf-8") as f:
                rules_data = json.load(f)
            self.rules_text.config(state=tk.NORMAL)
            if isinstance(rules_data, dict):
                for did, rules in rules_data.items():
                    self.rules_text.insert(tk.END, f"\n[{did}] ({len(rules)} rules)\n{'─'*40}\n")
                    for i, rule in enumerate(rules):
                        self.rules_text.insert(tk.END, f"  {i+1}. {rule}\n")
            elif isinstance(rules_data, list):
                for i, rule in enumerate(rules_data):
                    self.rules_text.insert(tk.END, f"  {i+1}. {rule}\n")
            self.rules_text.config(state=tk.DISABLED)

        # Output files tab
        if os.path.isdir(self.output_dir):
            for f in sorted(os.listdir(self.output_dir)):
                self.files_listbox.insert(tk.END, f"{self.output_dir}/{f}")

    def _open_selected_file(self, event):
        sel = self.files_listbox.curselection()
        if sel:
            path = self.files_listbox.get(sel[0])
            full = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            if os.path.exists(full):
                os.startfile(full)


if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()
