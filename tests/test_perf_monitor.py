"""Tests for LatencyMonitor and the @timed decorator in src/perf_monitor."""

from __future__ import annotations

import time
import unittest

from src.perf_monitor import (
    REALTIME_BUDGET_MS,
    WARNING_THRESHOLD_MS,
    LatencyMonitor,
    timed,
)
from src import perf_monitor as perf_monitor_module


class LatencyMonitorTests(unittest.TestCase):
    def test_record_and_avg_single_stage(self):
        m = LatencyMonitor()
        m.record("stage_a", 10.0)
        m.record("stage_a", 20.0)
        m.record("stage_a", 30.0)
        self.assertAlmostEqual(m.avg("stage_a"), 20.0, places=3)

    def test_avg_for_unknown_stage_is_zero(self):
        m = LatencyMonitor()
        self.assertEqual(m.avg("never_recorded"), 0.0)

    def test_rolling_window_caps_history(self):
        m = LatencyMonitor(window_size=3)
        for v in (100.0, 1.0, 2.0, 3.0):
            m.record("x", v)
        # Oldest (100.0) should be dropped — avg of the last 3.
        self.assertAlmostEqual(m.avg("x"), 2.0, places=3)

    def test_total_avg_sums_across_stages(self):
        m = LatencyMonitor()
        m.record("a", 5.0)
        m.record("b", 7.0)
        m.record("b", 9.0)  # avg(b) = 8.0
        self.assertAlmostEqual(m.total_avg(), 5.0 + 8.0, places=3)

    def test_reset_clears_all_state(self):
        m = LatencyMonitor()
        m.record("a", 5.0)
        m.record("b", 10.0)
        m.reset()
        self.assertEqual(m.avg("a"), 0.0)
        self.assertEqual(m.avg("b"), 0.0)
        self.assertEqual(m.total_avg(), 0.0)
        self.assertEqual(m.report()["stages"], {})

    def test_report_structure_and_thresholds(self):
        m = LatencyMonitor()
        m.record("detect", 10.0)
        m.record("score", 5.0)
        rep = m.report()
        self.assertEqual(rep["budget_ms"], REALTIME_BUDGET_MS)
        self.assertEqual(rep["warning_ms"], WARNING_THRESHOLD_MS)
        self.assertIn("detect", rep["stages"])
        self.assertIn("score", rep["stages"])
        self.assertAlmostEqual(rep["total_ms"], 15.0, places=2)
        self.assertFalse(rep["over_budget"])
        self.assertFalse(rep["over_warning"])
        # fps = 1000 / 15 ≈ 66.7
        self.assertAlmostEqual(rep["fps_estimate"], round(1000.0 / 15.0, 1), places=1)

    def test_report_flags_over_budget(self):
        m = LatencyMonitor()
        m.record("slow", REALTIME_BUDGET_MS + 5.0)
        rep = m.report()
        self.assertTrue(rep["over_budget"])
        self.assertTrue(rep["over_warning"])

    def test_report_flags_over_warning_only(self):
        m = LatencyMonitor()
        m.record("medium", WARNING_THRESHOLD_MS + 1.0)  # 26ms
        rep = m.report()
        self.assertTrue(rep["over_warning"])
        self.assertFalse(rep["over_budget"])

    def test_report_fps_zero_when_no_timings(self):
        m = LatencyMonitor()
        rep = m.report()
        self.assertEqual(rep["total_ms"], 0.0)
        self.assertEqual(rep["fps_estimate"], 0.0)


class TimedDecoratorTests(unittest.TestCase):
    def setUp(self):
        # Snapshot and clear the module-level singleton so tests are isolated.
        self._saved = perf_monitor_module.monitor
        perf_monitor_module.monitor = LatencyMonitor()

    def tearDown(self):
        perf_monitor_module.monitor = self._saved

    def test_timed_records_into_module_monitor(self):
        @timed("unit_test_stage")
        def work():
            time.sleep(0.005)  # ~5ms
            return 42

        result = work()
        self.assertEqual(result, 42)
        avg = perf_monitor_module.monitor.avg("unit_test_stage")
        # Sleep is best-effort, but it must be non-zero and at least a few ms.
        self.assertGreater(avg, 1.0)

    def test_timed_records_on_exception(self):
        @timed("raises")
        def boom():
            raise RuntimeError("fail")

        with self.assertRaises(RuntimeError):
            boom()
        # Even on exception, the timing should have been recorded.
        self.assertGreater(
            len(perf_monitor_module.monitor.timings.get("raises", [])), 0
        )

    def test_timed_preserves_function_metadata(self):
        @timed("meta")
        def my_fn(a, b):
            """docstring."""
            return a + b

        self.assertEqual(my_fn.__name__, "my_fn")
        self.assertEqual(my_fn.__doc__, "docstring.")
        self.assertEqual(my_fn(2, 3), 5)


if __name__ == "__main__":
    unittest.main()
