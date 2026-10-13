"""Mathematical engine for the MFAD Smart Parking project.

The engine deliberately exposes the linear-algebra pipeline used by the application:
1. Matrix representation of parking spots.
2. Vehicle demand vector.
3. Orthogonal projection of the demand vector into the span of available spot vectors.
4. Least-squares residual and distance based ranking.
5. Final assignment to the minimum-cost feasible spot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


SIZE_LEVEL = {"Compact": 1, "Standard": 2, "Large": 3}
FEATURE_NAMES = ["dx", "dy", "EV", "Accessible", "SizeCapacity"]


@dataclass
class ParkingSpot:
    spot_id: str
    row: int
    col: int
    x: float
    y: float
    ev: bool = False
    accessible: bool = False
    size_capacity: int = 2
    occupied: bool = False
    assigned_vehicle: Optional[str] = None


@dataclass
class Gate:
    gate_id: str
    x: float
    y: float


@dataclass
class Vehicle:
    vehicle_id: str
    entry_gate: str
    ev_required: bool = False
    accessible_required: bool = False
    size: str = "Standard"

    @property
    def size_level(self) -> int:
        return SIZE_LEVEL[self.size]


@dataclass
class CandidateScore:
    spot_id: str
    feasible: bool
    least_squares: float
    gate_distance: float
    lateral_offset: float
    total_cost: float
    reason: str = ""


@dataclass
class AssignmentResult:
    vehicle: Vehicle
    assigned_spot: Optional[ParkingSpot]
    candidates: List[CandidateScore]
    spot_matrix: np.ndarray
    vehicle_vector: np.ndarray
    projector: np.ndarray
    projected_vehicle_vector: np.ndarray
    log_lines: List[str] = field(default_factory=list)


class SmartParkingEngine:
    """Parking assignment engine with inspectable linear algebra."""

    def __init__(self, rows: int = 5, cols: int = 6, spacing: float = 2.2):
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        self.gates: Dict[str, Gate] = {
            "G1": Gate("G1", 0.0, 0.0),
            "G2": Gate("G2", (cols + 1) * spacing, (rows + 1) * spacing),
        }
        self.spots: List[ParkingSpot] = self._build_layout()

    def _build_layout(self) -> List[ParkingSpot]:
        spots: List[ParkingSpot] = []
        sid = 1
        for r in range(self.rows):
            for c in range(self.cols):
                # Create a visually mixed but deterministic set of facilities.
                ev = (sid % 7 == 0) or (sid in {5, 18, 20, 26, 7, 25, 30})
                accessible = c in {0, self.cols - 1} and r in {1, 2, 3}
                if sid in {4, 11, 18, 25, 30, 7}:
                    accessible = True
                size_capacity = 3 if (sid % 9 == 0 or sid in {7, 18, 25, 30}) else (2 if sid % 4 != 0 else 1)
                spots.append(
                    ParkingSpot(
                        spot_id=f"P{sid:02d}",
                        row=r,
                        col=c,
                        x=(c + 1) * self.spacing,
                        y=(r + 1) * self.spacing,
                        ev=ev,
                        accessible=accessible,
                        size_capacity=size_capacity,
                    )
                )
                sid += 1
        return spots

    def reset(self) -> None:
        for s in self.spots:
            s.occupied = False
            s.assigned_vehicle = None

    def get_gate(self, gate_id: str) -> Gate:
        if gate_id not in self.gates:
            raise ValueError(f"Unknown gate: {gate_id}")
        return self.gates[gate_id]

    def physical_position_matrix(self, gate_id: str) -> np.ndarray:
        gate = self.get_gate(gate_id)
        return np.array([[s.x - gate.x, s.y - gate.y] for s in self.spots], dtype=float)

    def build_feature_matrix(self, gate_id: str, spots: Optional[List[ParkingSpot]] = None) -> np.ndarray:
        """Return an N x 5 normalized feature matrix for selected spots."""
        gate = self.get_gate(gate_id)
        chosen = self.spots if spots is None else spots
        lot_w = max((self.cols + 1) * self.spacing, 1.0)
        lot_h = max((self.rows + 1) * self.spacing, 1.0)
        matrix = []
        for s in chosen:
            matrix.append(
                [
                    (s.x - gate.x) / lot_w,
                    (s.y - gate.y) / lot_h,
                    1.0 if s.ev else 0.0,
                    1.0 if s.accessible else 0.0,
                    s.size_capacity / 3.0,
                ]
            )
        return np.array(matrix, dtype=float)

    def build_vehicle_vector(self, vehicle: Vehicle) -> np.ndarray:
        # Spatial ideal = gate location itself, therefore dx=dy=0.
        return np.array(
            [
                0.0,
                0.0,
                1.0 if vehicle.ev_required else 0.0,
                1.0 if vehicle.accessible_required else 0.0,
                vehicle.size_level / 3.0,
            ],
            dtype=float,
        )

    @staticmethod
    def orthogonal_project(X: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project v onto the row-space of X using a Moore-Penrose pseudoinverse.

        P = X^T (X X^T)^+ X is an orthogonal projector in R^d.
        v_hat = P v is the least-squares closest point in that subspace.
        """
        if X.size == 0:
            d = v.shape[0]
            return np.zeros((d, d)), v.copy()
        gram = X @ X.T
        projector = X.T @ np.linalg.pinv(gram) @ X
        projected = projector @ v
        return projector, projected

    @staticmethod
    def gate_projection_metrics(spot: ParkingSpot, gate: Gate, rows: int, cols: int, spacing: float) -> Tuple[float, float]:
        """Compute Euclidean gate distance and lateral offset using projection.

        The lot-center direction defines the ideal approach direction from the gate.
        The perpendicular component is obtained with I - P_g.
        """
        p = np.array([spot.x - gate.x, spot.y - gate.y], dtype=float)
        center = np.array([
            ((cols + 1) * spacing) / 2 - gate.x,
            ((rows + 1) * spacing) / 2 - gate.y,
        ], dtype=float)
        if np.linalg.norm(center) < 1e-12:
            center = np.array([1.0, 0.0])
        g = center / np.linalg.norm(center)
        P_g = np.outer(g, g)
        parallel = P_g @ p
        perpendicular = (np.eye(2) - P_g) @ p
        return float(np.linalg.norm(p)), float(np.linalg.norm(perpendicular))

    def _eligibility_reason(self, vehicle: Vehicle, spot: ParkingSpot) -> str:
        if spot.occupied:
            return "Occupied"
        if vehicle.ev_required and not spot.ev:
            return "EV required"
        if vehicle.accessible_required and not spot.accessible:
            return "Accessible spot required"
        if spot.size_capacity < vehicle.size_level:
            return "Vehicle too large"
        return ""

    def assign(self, vehicle: Vehicle) -> AssignmentResult:
        gate = self.get_gate(vehicle.entry_gate)
        v = self.build_vehicle_vector(vehicle)

        eligible = []
        candidates: List[CandidateScore] = []
        for s in self.spots:
            reason = self._eligibility_reason(vehicle, s)
            if not reason:
                eligible.append(s)
            else:
                candidates.append(
                    CandidateScore(s.spot_id, False, np.inf, np.inf, np.inf, np.inf, reason)
                )

        if eligible:
            X = self.build_feature_matrix(vehicle.entry_gate, eligible)
            P, v_hat = self.orthogonal_project(X, v)
        else:
            X = np.empty((0, 5))
            P = np.zeros((5, 5))
            v_hat = v.copy()

        # Weighted least-squares distance. Position is slightly stronger than
        # facilities because the user explicitly asks for proximity to a gate.
        weights = np.array([1.15, 1.15, 1.60, 1.60, 1.25])
        ls_values = []
        gate_values = []
        lateral_values = []
        for s in eligible:
            row = self.build_feature_matrix(vehicle.entry_gate, [s])[0]
            residual = weights * (row - v_hat)
            ls = float(np.dot(residual, residual))
            gd, lat = self.gate_projection_metrics(s, gate, self.rows, self.cols, self.spacing)
            ls_values.append(ls)
            gate_values.append(gd)
            lateral_values.append(lat)

        if eligible:
            max_gate = max(gate_values) or 1.0
            max_lat = max(lateral_values) or 1.0
            for s, ls, gd, lat in zip(eligible, ls_values, gate_values, lateral_values):
                total = 0.60 * ls + 0.30 * (gd / max_gate) + 0.10 * (lat / max_lat)
                candidates.append(CandidateScore(s.spot_id, True, ls, gd, lat, total, "Eligible"))
            candidates_sorted = sorted(
                [c for c in candidates if c.feasible], key=lambda c: c.total_cost
            )
            winner_id = candidates_sorted[0].spot_id if candidates_sorted else None
        else:
            candidates_sorted = []
            winner_id = None

        assigned = next((s for s in self.spots if s.spot_id == winner_id), None)
        if assigned is not None:
            assigned.occupied = True
            assigned.assigned_vehicle = vehicle.vehicle_id

        top = sorted([c for c in candidates if c.feasible], key=lambda c: c.total_cost)[:4]
        logs = self._build_logs(vehicle, X, v, P, v_hat, candidates, assigned, top)
        return AssignmentResult(vehicle, assigned, candidates, X, v, P, v_hat, logs)

    def _fmt_matrix(self, M: np.ndarray, precision: int = 3) -> str:
        if M.size == 0:
            return "[]"
        return np.array2string(M, precision=precision, suppress_small=True)

    def _build_logs(
        self,
        vehicle: Vehicle,
        X: np.ndarray,
        v: np.ndarray,
        P: np.ndarray,
        v_hat: np.ndarray,
        candidates: List[CandidateScore],
        assigned: Optional[ParkingSpot],
        top: List[CandidateScore],
    ) -> List[str]:
        logs: List[str] = []
        logs.append("=== SMART PARKING LINEAR-ALGEBRA PIPELINE ===")
        logs.append(f"Vehicle: {vehicle.vehicle_id} | Gate: {vehicle.entry_gate} | EV={vehicle.ev_required} | Accessible={vehicle.accessible_required} | Size={vehicle.size}")
        logs.append("")
        logs.append("STEP 1 - RAW / CANDIDATE FEATURE MATRIX X (rows = feasible spots)")
        logs.append("Columns = [dx, dy, EV, Accessible, SizeCapacity]")
        logs.append(self._fmt_matrix(X))
        logs.append("")
        logs.append("STEP 2 - VEHICLE DEMAND VECTOR v")
        logs.append(self._fmt_matrix(v))
        logs.append("")
        logs.append("STEP 3 - ORTHOGONAL PROJECTION")
        logs.append("P = X^T (X X^T)^+ X")
        logs.append(self._fmt_matrix(P))
        logs.append("Projected demand v_hat = P v")
        logs.append(self._fmt_matrix(v_hat))
        logs.append("")
        logs.append("STEP 4 - LEAST-SQUARES RANKING")
        logs.append("LS_i = || W (x_i - v_hat) ||_2^2")
        logs.append("Total_i = 0.60*LS_i + 0.30*normalized_gate_distance + 0.10*normalized_lateral_offset")
        for rank, c in enumerate(top, 1):
            logs.append(
                f"#{rank} {c.spot_id}: LS={c.least_squares:.4f}, gate={c.gate_distance:.2f}, lateral={c.lateral_offset:.2f}, total={c.total_cost:.4f}"
            )
        logs.append("")
        if assigned:
            logs.append(f"FINAL ASSIGNMENT -> {assigned.spot_id} (row={assigned.row+1}, col={assigned.col+1})")
        else:
            logs.append("FINAL ASSIGNMENT -> No feasible spot available")
        logs.append("=== END PIPELINE ===")
        return logs

    def occupancy_summary(self) -> Dict[str, int]:
        total = len(self.spots)
        occupied = sum(s.occupied for s in self.spots)
        ev = sum(s.ev for s in self.spots)
        accessible = sum(s.accessible for s in self.spots)
        return {
            "total": total,
            "occupied": occupied,
            "available": total - occupied,
            "ev_total": ev,
            "accessible_total": accessible,
        }
