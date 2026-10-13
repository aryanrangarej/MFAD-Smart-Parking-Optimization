"""Interactive Tkinter GUI for the Smart Parking MFAD mini-project."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk, filedialog

# Allow running directly from the src/ folder or project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parking_engine import SmartParkingEngine, Vehicle  # noqa: E402


class SmartParkingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MFAD 2026 | Smart Parking using Linear Algebra")
        self.geometry("1220x760")
        self.minsize(1100, 680)
        self.engine = SmartParkingEngine(rows=5, cols=6)
        self.vehicle_counter = 1
        self.last_result = None
        self.status_var = tk.StringVar(value="Ready for a vehicle.")
        self.summary_var = tk.StringVar()
        self._build_ui()
        self._refresh_grid()
        self._update_summary()

    def _build_ui(self):
        header = tk.Frame(self, bg="#152238", height=74)
        header.pack(fill="x")
        tk.Label(header, text="SMART PARKING", bg="#152238", fg="white",
                 font=("Segoe UI", 22, "bold")).pack(side="left", padx=22, pady=(10, 0))
        tk.Label(header, text="Matrix Representation • Orthogonal Projection • Least-Squares Optimization",
                 bg="#152238", fg="#bcd0e5", font=("Segoe UI", 10)).pack(side="left", padx=10, pady=(18, 0))

        main = tk.Frame(self, bg="#eef2f7")
        main.pack(fill="both", expand=True)

        control = tk.Frame(main, bg="white", bd=1, relief="solid")
        control.pack(side="left", fill="y", padx=14, pady=14)
        tk.Label(control, text="Incoming Vehicle", bg="white", fg="#152238",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=16, pady=(14, 8))

        form = tk.Frame(control, bg="white")
        form.pack(fill="x", padx=16)
        self.gate_var = tk.StringVar(value="G1")
        self.ev_var = tk.BooleanVar(value=False)
        self.access_var = tk.BooleanVar(value=False)
        self.size_var = tk.StringVar(value="Standard")

        self._field(form, "Entry gate", ttk.Combobox(form, textvariable=self.gate_var,
                                                      values=list(self.engine.gates), state="readonly", width=16))
        tk.Checkbutton(form, text="EV charging required", variable=self.ev_var, bg="white").pack(anchor="w", pady=7)
        tk.Checkbutton(form, text="Accessible spot required", variable=self.access_var, bg="white").pack(anchor="w", pady=7)
        self._field(form, "Vehicle size", ttk.Combobox(form, textvariable=self.size_var,
                                                        values=["Compact", "Standard", "Large"], state="readonly", width=16))

        btns = tk.Frame(control, bg="white")
        btns.pack(fill="x", padx=16, pady=12)
        tk.Button(btns, text="ASSIGN OPTIMAL SPOT", command=self.assign_vehicle,
                  bg="#1976d2", fg="white", relief="flat", padx=10, pady=9,
                  font=("Segoe UI", 10, "bold")).pack(fill="x", pady=4)
        tk.Button(btns, text="AUTO-DEMO (5 VEHICLES)", command=self.auto_demo,
                  bg="#455a64", fg="white", relief="flat", padx=10, pady=8).pack(fill="x", pady=4)
        tk.Button(btns, text="RESET LOT", command=self.reset_lot,
                  bg="#eceff1", fg="#263238", relief="flat", padx=10, pady=8).pack(fill="x", pady=4)
        tk.Button(btns, text="EXPORT LOG", command=self.export_log,
                  bg="#eceff1", fg="#263238", relief="flat", padx=10, pady=8).pack(fill="x", pady=4)

        tk.Label(control, textvariable=self.summary_var, bg="#f7f9fc", fg="#364a63",
                 justify="left", anchor="w", font=("Segoe UI", 9), padx=12, pady=12).pack(fill="x", padx=16, pady=(8, 4))
        tk.Label(control, text="Hard constraints are checked first.\nThen linear-algebra scoring ranks\nthe remaining feasible spots.",
                 bg="white", fg="#6b7785", justify="left", font=("Segoe UI", 9)).pack(anchor="w", padx=16, pady=12)

        center = tk.Frame(main, bg="#eef2f7")
        center.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=14)
        tk.Label(center, text="Parking Grid", bg="#eef2f7", fg="#152238",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        grid_card = tk.Frame(center, bg="white", bd=1, relief="solid")
        grid_card.pack(fill="both", expand=False, padx=2, pady=1)
        self.grid_frame = tk.Frame(grid_card, bg="white")
        self.grid_frame.pack(padx=10, pady=10)

        # Use a real color legend instead of one gray text label.
        self.legend = tk.Frame(center, bg="#eef2f7")
        self.legend.pack(anchor="w", pady=8)
        legend_items = [
            ("#43a047", "Available"),
            ("#1976d2", "EV"),
            ("#ff9800", "Accessible"),
            ("#e53935", "Occupied"),
            ("#8e44ad", "EV + Accessible"),
        ]
        for color, label in legend_items:
            item = tk.Frame(self.legend, bg="#eef2f7")
            item.pack(side="left", padx=(0, 12))
            tk.Label(item, text="  ", bg=color, width=2, height=1,
                     relief="solid", bd=1).pack(side="left", padx=(0, 4))
            tk.Label(item, text=label, bg="#eef2f7", fg="#536273",
                     font=("Segoe UI", 9)).pack(side="left")
        self.status_label = tk.Label(center, textvariable=self.status_var, bg="#ffffff", fg="#152238",
                                     anchor="w", padx=12, pady=10, font=("Segoe UI", 10, "bold"))
        self.status_label.pack(fill="x", pady=(3, 8))

        right = tk.Frame(main, bg="white", bd=1, relief="solid")
        right.pack(side="right", fill="both", expand=True, padx=(0, 14), pady=14)
        tk.Label(right, text="Pipeline Visualizer & Output Log", bg="white", fg="#152238",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 6))
        text_frame = tk.Frame(right, bg="white")
        text_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.log_text = tk.Text(text_frame, wrap="none", bg="#0f172a", fg="#dbeafe",
                                insertbackground="white", font=("Consolas", 9))
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=yscroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

    def _field(self, parent, label, widget):
        tk.Label(parent, text=label, bg="white", fg="#546273", font=("Segoe UI", 9)).pack(anchor="w", pady=(7, 2))
        widget.pack(anchor="w")

    def _spot_color(self, spot):
        if spot.occupied:
            return "#e53935"
        if spot.ev and spot.accessible:
            return "#8e44ad"
        if spot.accessible:
            return "#ff9800"
        if spot.ev:
            return "#1976d2"
        return "#43a047"

    def _refresh_grid(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        for s in self.engine.spots:
            b = tk.Button(self.grid_frame, text=s.spot_id, width=9, height=3,
                          bg=self._spot_color(s), fg="white", activebackground=self._spot_color(s),
                          relief="flat", font=("Segoe UI", 9, "bold"),
                          command=lambda sid=s.spot_id: self.show_spot(sid))
            b.grid(row=s.row, column=s.col, padx=4, pady=4)
        self._update_summary()

    def _update_summary(self):
        d = self.engine.occupancy_summary()
        self.summary_var.set(
            f"Total spots: {d['total']}\n"
            f"Available: {d['available']}\n"
            f"Occupied: {d['occupied']}\n"
            f"EV-capable: {d['ev_total']}\n"
            f"Accessible: {d['accessible_total']}"
        )

    def show_spot(self, sid):
        s = next(x for x in self.engine.spots if x.spot_id == sid)
        details = (f"{s.spot_id}\nCoordinates: ({s.x:.1f}, {s.y:.1f})\n"
                   f"EV: {'Yes' if s.ev else 'No'}\n"
                   f"Accessible: {'Yes' if s.accessible else 'No'}\n"
                   f"Capacity: {s.size_capacity}\n"
                   f"Status: {'Occupied by ' + s.assigned_vehicle if s.occupied else 'Available'}")
        messagebox.showinfo("Parking Spot", details)

    def _append_log(self, lines):
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "\n".join(lines))
        self.log_text.see("1.0")

    def assign_vehicle(self, vehicle_id=None):
        if vehicle_id is None:
            vehicle_id = f"V{self.vehicle_counter:02d}"
            self.vehicle_counter += 1
        vehicle = Vehicle(vehicle_id, self.gate_var.get(), self.ev_var.get(), self.access_var.get(), self.size_var.get())
        result = self.engine.assign(vehicle)
        self.last_result = result
        self._append_log(result.log_lines)
        self._refresh_grid()
        if result.assigned_spot:
            self.status_var.set(f"{vehicle.vehicle_id} → {result.assigned_spot.spot_id}. Assignment chosen by minimum composite cost.")
        else:
            self.status_var.set(f"{vehicle.vehicle_id} could not be assigned: no feasible spot.")
        return result

    def auto_demo(self):
        self.reset_lot(log=False)
        demo = [
            ("G1", False, False, "Compact"),
            ("G1", True, False, "Standard"),
            ("G2", False, True, "Standard"),
            ("G2", True, True, "Large"),
            ("G1", False, False, "Large"),
        ]
        last = None
        all_lines = ["AUTO-DEMO: five vehicles enter sequentially.\n"]
        for idx, (gate, ev, acc, size) in enumerate(demo, start=1):
            self.gate_var.set(gate); self.ev_var.set(ev); self.access_var.set(acc); self.size_var.set(size)
            v = Vehicle(f"D{idx:02d}", gate, ev, acc, size)
            last = self.engine.assign(v)
            all_lines.extend(last.log_lines)
            all_lines.append("")
        self.last_result = last
        self._append_log(all_lines)
        self._refresh_grid()
        self.status_var.set("Auto-demo complete. Five sample vehicles were processed.")

    def reset_lot(self, log=True):
        self.engine.reset()
        self.vehicle_counter = 1
        self.last_result = None
        self._refresh_grid()
        self.status_var.set("Lot reset. Ready for a vehicle.")
        if log:
            self.log_text.delete("1.0", "end")
            self.log_text.insert("end", "Lot reset. Run an assignment or AUTO-DEMO to generate the pipeline log.")

    def export_log(self):
        content = self.log_text.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Export Log", "There is no log to export yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            Path(path).write_text(content, encoding="utf-8")
            messagebox.showinfo("Export Log", f"Saved to:\n{path}")


if __name__ == "__main__":
    app = SmartParkingApp()
    app.mainloop()
