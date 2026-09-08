"""Tests for the sharing- and session-aware request scheduler.

Run with:
    PYTHONPATH=src python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import asyncio

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from kvcache.session import Session, Turn  # noqa: E402
from kvcache.overlap import OverlapIndex, ngrams, ngram_id, detect_shared_ngrams  # noqa: E402
from kvcache.workload import WorkloadConfig, generate_workload  # noqa: E402
from kvcache.policies import (  # noqa: E402
    FIFOPolicy,
    SessionAwarePolicy,
    SharingAwarePolicy,
    CombinedPolicy,
    QueuedRequest,
)
from kvcache.bench import BenchConfig, MockVLLMBackend, run_benchmark  # noqa: E402
from kvcache.vllm_backend import BackendConfig, tokens_to_text  # noqa: E402
from kvcache.metrics import MetricsLogger, MetricsConfig, RequestRecord  # noqa: E402


class TestSession(unittest.TestCase):
    def test_session_grows(self):
        s = Session(session_id="s0", start_t=0.0, state="ACTIVE")
        for i in range(3):
            s.turns.append(Turn(turn_index=i, t=i * 0.5, tokens=(1, 2, 3, 4), role="user"))
            s.last_turn_t = i * 0.5
        self.assertEqual(s.turn_count, 3)
        self.assertEqual(len(s.context_tokens), 12)

    def test_return_likelihood_in_unit_range(self):
        s = Session(session_id="s0", start_t=0.0, state="ACTIVE")
        s.turns.append(Turn(0, 0.0, (1, 2), "user"))
        s.last_turn_t = 0.0
        rl = s.return_likelihood(now=0.0)
        self.assertGreaterEqual(rl, 0.0)
        self.assertLessEqual(rl, 1.0)


class TestOverlap(unittest.TestCase):
    def test_ngram_id_is_stable(self):
        a = ngram_id((1, 2, 3, 4, 5, 6, 7, 8))
        b = ngram_id((1, 2, 3, 4, 5, 6, 7, 8))
        self.assertEqual(a, b)

    def test_detect_shared_ngrams_prefix_overlap(self):
        a = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        b = (1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99)
        self.assertGreaterEqual(detect_shared_ngrams(a, b, n=8), 1)

    def test_detect_shared_ngrams_mid_overlap(self):
        a = (0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        b = (99, 99, 99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 99, 99)
        self.assertGreaterEqual(detect_shared_ngrams(a, b, n=8), 1)

    def test_overlap_index_other_refs(self):
        idx = OverlapIndex(n=8)
        a = (0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        idx.touch_session("s0", a)
        b = (99, 99, 99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 99, 99)
        others = idx.has_other_refs("s1", b)
        self.assertIn("s0", others)
        self.assertNotIn("s1", others)


class TestWorkload(unittest.TestCase):
    def test_workload_is_deterministic(self):
        cfg = WorkloadConfig(num_sessions=5, sim_window_s=60, seed=42)
        a = generate_workload(cfg)
        b = generate_workload(cfg)
        self.assertEqual(len(a.events), len(b.events))
        self.assertEqual(
            [e.t for e in a.events],
            [e.t for e in b.events],
        )

    def test_sharing_fraction(self):
        cfg = WorkloadConfig(
            num_sessions=20, sim_window_s=60, seed=1, overlap_fraction=0.5
        )
        w = generate_workload(cfg)
        self.assertGreater(len(w.sharing_sids), 5)
        self.assertLess(len(w.sharing_sids), 15)

    def test_events_sorted(self):
        w = generate_workload(WorkloadConfig(num_sessions=5, sim_window_s=30, seed=1))
        ts = [e.t for e in w.events]
        self.assertEqual(ts, sorted(ts))


class TestPolicies(unittest.TestCase):
    def test_fifo_preserves_order(self):
        w = generate_workload(WorkloadConfig(num_sessions=3, sim_window_s=30, seed=1))
        events = w.events[:5]
        queue = [QueuedRequest(event=e, arrival_t=e.t, enqueue_seq=i) for i, e in enumerate(events)]
        d = FIFOPolicy().score_queue(queue, {s.session_id: s for s in w.sessions}, OverlapIndex(n=8), 0.0)
        out_ts = [q.event.t for q in d.ordered]
        self.assertEqual(out_ts, sorted(out_ts))

    def test_policies_produce_different_orderings(self):
        s_sharer = Session(session_id="sharer", start_t=0.0, state="ACTIVE")
        s_sharer.turns.append(
            Turn(0, 0.0, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), "user")
        )
        s_sharer.last_turn_t = 0.0
        s_solo = Session(session_id="solo", start_t=0.0, state="ACTIVE")
        for i in range(15):
            s_solo.turns.append(Turn(i, 0.0, (i + 100,) * 8, "user"))
        s_solo.last_turn_t = 0.0
        sess_map = {"sharer": s_sharer, "solo": s_solo}

        oi = OverlapIndex(n=8)
        oi.touch_session("earlier", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20))

        sharer_tokens = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        solo_tokens = tuple(range(100, 100 + 8))

        from kvcache.workload import TurnEvent
        ev_sharer = TurnEvent(session_id="sharer", turn_index=1, t=0.0, tokens=sharer_tokens, role="user", is_active=True)
        ev_solo = TurnEvent(session_id="solo", turn_index=15, t=0.0, tokens=solo_tokens, role="user", is_active=True)

        queue = [
            QueuedRequest(event=ev_sharer, arrival_t=0.0, enqueue_seq=0),
            QueuedRequest(event=ev_solo, arrival_t=0.0, enqueue_seq=1),
        ]
        sa = SessionAwarePolicy().score_queue(queue, sess_map, oi, 0.0)
        sh = SharingAwarePolicy().score_queue(queue, sess_map, oi, 0.0)
        self.assertEqual(sa.ordered[0].event.session_id, "solo")
        self.assertEqual(sh.ordered[0].event.session_id, "sharer")


class TestBenchIntegration(unittest.TestCase):
    def test_benchmark_runs_all_four_policies(self):
        async def go():
            w = generate_workload(
                WorkloadConfig(num_sessions=3, sim_window_s=8, seed=1, mean_idle_gap_s=2)
            )
            with tempfile.TemporaryDirectory() as tmp:
                for policy in ["fifo", "session-aware", "sharing-aware", "combined"]:
                    backend = MockVLLMBackend(BackendConfig(max_num_seqs=2))
                    await backend.start()
                    cfg = BenchConfig(
                        policy_name=policy,
                        combined_alpha=0.5,
                        backend=BackendConfig(max_num_seqs=2),
                        max_new_tokens=4,
                        sla_latency_ms=2000.0,
                        hit_latency_threshold_ms=300.0,
                        output_dir=tmp,
                        run_label=f"smoke_{policy}",
                        speed_factor=300.0,
                    )
                    r = await run_benchmark(w, cfg, backend=backend)
                    self.assertGreater(r.n_records, 0)
                    await backend.stop()
                for policy in ["fifo", "session-aware", "sharing-aware", "combined"]:
                    self.assertTrue(
                        os.path.exists(os.path.join(tmp, f"time_series_smoke_{policy}.csv"))
                    )
                    self.assertTrue(
                        os.path.exists(os.path.join(tmp, f"per_session_smoke_{policy}.csv"))
                    )
                    self.assertTrue(
                        os.path.exists(os.path.join(tmp, f"summary_smoke_{policy}.csv"))
                    )
        asyncio.run(go())

    def test_tokens_to_text_deterministic(self):
        a = tokens_to_text((0, 1, 2, 3, 4), 4000)
        b = tokens_to_text((0, 1, 2, 3, 4), 4000)
        self.assertEqual(a, b)
        self.assertEqual(len(a.split()), 5)


class TestMetrics(unittest.TestCase):
    def test_metrics_logger_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = MetricsLogger(MetricsConfig(output_dir=tmp, run_label="x"))
            for i in range(3):
                m.record(
                    RequestRecord(
                        request_id=f"r{i}",
                        session_id="s0",
                        turn_index=i,
                        submit_t=i * 5.0,
                        complete_t=i * 5.0 + 0.1,
                        latency_ms=100,
                        n_prompt_tokens=20,
                        n_output_tokens=10,
                        hit=True,
                        shared=False,
                        policy_name="fifo",
                        capacity_setting="constrained",
                        in_flight_at_submit=1,
                    )
                )
            m.flush_window(0)
            s = m.finalize(sim_window_s=30.0)
            self.assertEqual(s["lookups"], 3)
            self.assertTrue(os.path.exists(os.path.join(tmp, "time_series_x.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "per_session_x.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "summary_x.csv")))


class TestAnalysisLabels(unittest.TestCase):
    def test_split_base_labels(self):
        from kvcache.analysis import _split_policy_and_capacity  # noqa: E402
        self.assertEqual(
            _split_policy_and_capacity("combined_constrained"),
            ("combined", "constrained", ""),
        )
        self.assertEqual(
            _split_policy_and_capacity("session-aware_generous"),
            ("session-aware", "generous", ""),
        )

    def test_split_variant_labels(self):
        from kvcache.analysis import _split_policy_and_capacity  # noqa: E402
        self.assertEqual(
            _split_policy_and_capacity("combined_constrained_a025"),
            ("combined", "constrained", "_a025"),
        )
        self.assertEqual(
            _split_policy_and_capacity("fifo_generous_s1"),
            ("fifo", "generous", "_s1"),
        )

    def test_ensure_base_results_restores_once(self):
        import tempfile  # noqa: E402
        from kvcache.analysis import ensure_base_results  # noqa: E402
        with tempfile.TemporaryDirectory() as repo:
            arch = os.path.join(repo, "result_from_first_experiment", "csv")
            os.makedirs(arch)
            with open(os.path.join(arch, "summary_fifo_constrained.csv"), "w") as f:
                f.write("policy,hit_rate\nfifo_constrained,0.5\n")
            csv_dir = os.path.join(repo, "results", "csv")
            self.assertEqual(ensure_base_results(csv_dir), 1)
            self.assertTrue(os.path.exists(
                os.path.join(csv_dir, "summary_fifo_constrained.csv")))
            # second call is a no-op (dir no longer empty of summaries)
            self.assertEqual(ensure_base_results(csv_dir), 0)


if __name__ == "__main__":
    unittest.main()

