import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parking_engine import SmartParkingEngine, Vehicle  # noqa: E402


class TestSmartParking(unittest.TestCase):
    def test_projection_is_approximately_idempotent(self):
        engine = SmartParkingEngine()
        X = engine.build_feature_matrix("G1", engine.spots[:10])
        v = engine.build_vehicle_vector(Vehicle("V", "G1", True, False, "Standard"))
        P, _ = engine.orthogonal_project(X, v)
        self.assertTrue(np.allclose(P @ P, P, atol=1e-7))
        self.assertTrue(np.allclose(P.T, P, atol=1e-7))

    def test_ev_vehicle_gets_ev_spot(self):
        engine = SmartParkingEngine()
        result = engine.assign(Vehicle("V01", "G1", True, False, "Standard"))
        self.assertIsNotNone(result.assigned_spot)
        self.assertTrue(result.assigned_spot.ev)
        self.assertGreaterEqual(result.assigned_spot.size_capacity, 2)

    def test_accessible_large_vehicle(self):
        engine = SmartParkingEngine()
        result = engine.assign(Vehicle("V01", "G2", False, True, "Large"))
        self.assertIsNotNone(result.assigned_spot)
        self.assertTrue(result.assigned_spot.accessible)
        self.assertEqual(result.assigned_spot.size_capacity, 3)

    def test_main_demo_has_multiple_feasible_candidates(self):
        engine = SmartParkingEngine()
        result = engine.assign(Vehicle("DEMO-01", "G2", True, True, "Large"))
        feasible = [c for c in result.candidates if c.feasible]
        self.assertGreaterEqual(len(feasible), 3)
        self.assertEqual(result.assigned_spot.spot_id, "P30")

    def test_occupied_spot_is_not_reused(self):
        engine = SmartParkingEngine()
        first = engine.assign(Vehicle("V01", "G1", True, False, "Standard"))
        second = engine.assign(Vehicle("V02", "G1", True, False, "Standard"))
        self.assertNotEqual(first.assigned_spot.spot_id, second.assigned_spot.spot_id)

    def test_results_are_reproducible(self):
        vehicle = Vehicle("DEMO-01", "G2", True, True, "Large")
        first = TestSmartParking._run_clean_assignment(vehicle)
        second = TestSmartParking._run_clean_assignment(vehicle)
        self.assertEqual(first[0], second[0])
        self.assertAlmostEqual(first[1], second[1], places=10)

    @staticmethod
    def _run_clean_assignment(vehicle):
        engine = SmartParkingEngine()
        result = engine.assign(vehicle)
        return result.assigned_spot.spot_id, result.candidates[[c.spot_id for c in result.candidates].index(result.assigned_spot.spot_id)].total_cost


if __name__ == "__main__":
    unittest.main()
