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
    @staticmethod
    def _rec(i, submit_t, n_prompt=20, n_cached=0, hit=True, shared=False,
             latency_ms=100):
        return RequestRecord(
            request_id=f"r{i}",
            session_id="s0",
            turn_index=i,
            submit_t=submit_t,
            complete_t=submit_t + 0.1,
            latency_ms=latency_ms,
            n_prompt_tokens=n_prompt,
            n_output_tokens=10,
            hit=hit,
            shared=shared,
            policy_name="fifo",
            capacity_setting="constrained",
            in_flight_at_submit=1,
            num_cached_tokens=n_cached,
            hit_basis="cached_tokens",
            proxy_hit=latency_ms <= 300,
        )

    def test_metrics_logger_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = MetricsLogger(
                MetricsConfig(output_dir=tmp, run_label="x",
                              discard_warmup_windows=0)
            )
            for i in range(3):
                m.record(self._rec(i, i * 5.0, n_cached=10))
            m.flush_window(0)
            s = m.finalize(sim_window_s=30.0)
            self.assertEqual(s["lookups"], 3)
            self.assertTrue(os.path.exists(os.path.join(tmp, "time_series_x.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "per_session_x.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "summary_x.csv")))

    def test_cached_token_rate_is_token_weighted(self):
        """Headline metric must weight by tokens, not average per-request."""
        with tempfile.TemporaryDirectory() as tmp:
            m = MetricsLogger(
                MetricsConfig(output_dir=tmp, run_label="tw",
                              discard_warmup_windows=0)
            )
            # One long request with no reuse, one short one fully reused.
            # Token-weighted rate = 10 / 1010, NOT the 0.5 you would get
            # by averaging per-request fractions.
            m.record(self._rec(0, 0.0, n_prompt=1000, n_cached=0))
            m.record(self._rec(1, 1.0, n_prompt=10, n_cached=10))
            s = m.finalize(sim_window_s=30.0)
            self.assertEqual(s["prompt_tokens"], 1010)
            self.assertEqual(s["cached_tokens"], 10)
            self.assertAlmostEqual(s["cached_token_rate"], 10 / 1010, places=6)
            self.assertEqual(s["hit_basis"], "cached_tokens")
            self.assertAlmostEqual(s["cache_ground_truth_coverage"], 1.0)

    def test_warmup_windows_excluded_from_summary(self):
        """Window 0 must not be allowed to set the headline P99."""
        with tempfile.TemporaryDirectory() as tmp:
            m = MetricsLogger(
                MetricsConfig(output_dir=tmp, run_label="w",
                              discard_warmup_windows=1, window_s=30.0)
            )
            # One pathological warmup request (model load) in window 0...
            m.record(self._rec(0, 1.0, latency_ms=56_000))
            # ...and clean steady-state traffic in window 1.
            for i in range(1, 11):
                m.record(self._rec(i, 30.0 + i, latency_ms=700))
            s = m.finalize(sim_window_s=60.0)
            self.assertEqual(s["lookups"], 10)
            self.assertEqual(s["n_records_all"], 11)
            self.assertEqual(s["n_warmup_windows_discarded"], 1)
            self.assertLess(s["p99_latency_ms"], 1000)
            # The warmup window is still written out, just flagged.
            ts = open(os.path.join(tmp, "time_series_w.csv")).read().splitlines()
            self.assertIn("is_warmup", ts[0])
            self.assertEqual(ts[1].split(",")[3], "1")
            self.assertEqual(ts[2].split(",")[3], "0")

    def test_proxy_hit_rate_reported_separately(self):
        """The old latency metric is published beside ground truth, not as it."""
        with tempfile.TemporaryDirectory() as tmp:
            m = MetricsLogger(
                MetricsConfig(output_dir=tmp, run_label="p",
                              discard_warmup_windows=0,
                              hit_latency_threshold_ms=300)
            )
            # Every request reused nothing (true miss) but came back fast,
            # so the latency proxy would have called all of them hits.
            for i in range(4):
                m.record(self._rec(i, i * 1.0, n_cached=0, hit=False,
                                   latency_ms=100))
            s = m.finalize(sim_window_s=30.0)
            self.assertEqual(s["cached_token_rate"], 0.0)
            self.assertEqual(s["hit_rate"], 0.0)
            self.assertEqual(s["proxy_hit_rate"], 1.0)


class TestCacheGroundTruth(unittest.TestCase):
    """The fix that matters: hit/miss comes from vLLM, not from latency."""

    def test_extractor_prefers_request_output_field(self):
        from kvcache.vllm_backend import extract_num_cached_tokens

        class Out:
            num_cached_tokens = 48

        self.assertEqual(
            extract_num_cached_tokens(Out()),
            (48, "RequestOutput.num_cached_tokens"),
        )

    def test_extractor_falls_back_through_metrics(self):
        from kvcache.vllm_backend import extract_num_cached_tokens

        class M:
            num_cached_tokens = 16

        class Out:
            metrics = M()

        n, src = extract_num_cached_tokens(Out())
        self.assertEqual(n, 16)
        self.assertIn("metrics", src)

    def test_extractor_returns_minus_one_when_absent(self):
        """Must refuse to guess. -1 is what marks a run uninterpretable."""
        from kvcache.vllm_backend import extract_num_cached_tokens

        class Out:
            outputs = []
            metrics = None

        self.assertEqual(extract_num_cached_tokens(Out()), (-1, ""))

    def test_request_result_cached_fraction(self):
        from kvcache.vllm_backend import RequestResult

        r = RequestResult(
            request_id="r", text="", submit_t=0.0, submit_wall_t=0.0,
            first_token_t=0.0, complete_t=0.0, n_output_tokens=4,
            n_prompt_tokens=200, num_cached_tokens=50,
        )
        self.assertTrue(r.has_cache_ground_truth)
        self.assertAlmostEqual(r.cached_fraction, 0.25)

        miss = RequestResult(
            request_id="r", text="", submit_t=0.0, submit_wall_t=0.0,
            first_token_t=0.0, complete_t=0.0, n_output_tokens=4,
            n_prompt_tokens=200, num_cached_tokens=0,
        )
        self.assertTrue(miss.has_cache_ground_truth)
        self.assertEqual(miss.cached_fraction, 0.0)

        unknown = RequestResult(
            request_id="r", text="", submit_t=0.0, submit_wall_t=0.0,
            first_token_t=0.0, complete_t=0.0, n_output_tokens=4,
            n_prompt_tokens=200,
        )
        self.assertFalse(unknown.has_cache_ground_truth)

    def test_mock_prefix_cache_reuses_repeated_prefix(self):
        from kvcache.bench import MockVLLMBackend
        from kvcache.vllm_backend import BackendConfig

        b = MockVLLMBackend(BackendConfig(gpu_memory_utilization=0.7))
        prompt = " ".join(f"w{i%37}" for i in range(320))
        first, n = b._lookup_and_insert(prompt)
        self.assertEqual(first, 0)  # cold: nothing to reuse
        second, _ = b._lookup_and_insert(prompt)
        self.assertGreater(second, 0)  # warm: prefix is resident
        self.assertLessEqual(second, n)

    def test_mock_prefix_cache_evicts_under_pressure(self):
        """Constrained capacity must actually lose blocks. The lever has to bite."""
        from kvcache.bench import MockVLLMBackend
        from kvcache.vllm_backend import BackendConfig

        def reuse_after_churn(gpu_mem):
            b = MockVLLMBackend(BackendConfig(gpu_memory_utilization=gpu_mem))
            target = " ".join(f"t{i}" for i in range(320))
            b._lookup_and_insert(target)
            # Churn lots of unrelated traffic through the pool.
            for k in range(60):
                b._lookup_and_insert(" ".join(f"x{k}_{i}" for i in range(320)))
            again, _ = b._lookup_and_insert(target)
            return again, b.n_evictions

        tight, eviction_tight = reuse_after_churn(0.02)
        roomy, _ = reuse_after_churn(0.9)
        self.assertGreater(eviction_tight, 0)
        self.assertEqual(tight, 0)         # evicted under pressure
        self.assertGreater(roomy, 0)       # survived with headroom


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

    @staticmethod
    def _make_archive(repo, body):
        arch = os.path.join(repo, "result_from_first_experiment", "csv")
        os.makedirs(arch, exist_ok=True)
        with open(os.path.join(arch, "summary_fifo_constrained.csv"), "w") as f:
            f.write(body)
        return os.path.join(repo, "results", "csv")

    def test_ensure_base_results_restores_ground_truth_archive(self):
        import tempfile  # noqa: E402
        from kvcache.analysis import ensure_base_results  # noqa: E402
        with tempfile.TemporaryDirectory() as repo:
            csv_dir = self._make_archive(
                repo,
                "policy,hit_rate,hit_basis\n"
                "fifo_constrained,0.5,cached_tokens\n",
            )
            self.assertEqual(ensure_base_results(csv_dir), 1)
            self.assertTrue(os.path.exists(
                os.path.join(csv_dir, "summary_fifo_constrained.csv")))
            # second call is a no-op (dir no longer empty of summaries)
            self.assertEqual(ensure_base_results(csv_dir), 0)

    def test_ensure_base_results_refuses_legacy_archive(self):
        """A latency-proxy archive must NOT be restored.

        If it were, the runner's "skip finished labels" check would see
        summary_<label>.csv and skip the whole new matrix, and the analysis
        would report the superseded numbers as if they were fresh.
        """
        import tempfile  # noqa: E402
        from kvcache.analysis import ensure_base_results  # noqa: E402
        with tempfile.TemporaryDirectory() as repo:
            csv_dir = self._make_archive(
                repo, "policy,hit_rate\nfifo_constrained,0.5\n"
            )
            self.assertEqual(ensure_base_results(csv_dir), 0)
            self.assertFalse(os.path.exists(
                os.path.join(csv_dir, "summary_fifo_constrained.csv")))

    def test_summary_has_ground_truth(self):
        import tempfile  # noqa: E402
        from kvcache.analysis import summary_has_ground_truth  # noqa: E402
        with tempfile.TemporaryDirectory() as d:
            gt = os.path.join(d, "gt.csv")
            with open(gt, "w") as f:
                f.write("policy,hit_basis\nx,cached_tokens\n")
            self.assertTrue(summary_has_ground_truth(gt))

            old = os.path.join(d, "old.csv")
            with open(old, "w") as f:
                f.write("policy,hit_rate\nx,0.5\n")
            self.assertFalse(summary_has_ground_truth(old))

            fallback = os.path.join(d, "fb.csv")
            with open(fallback, "w") as f:
                f.write("policy,hit_basis\nx,latency_proxy\n")
            self.assertFalse(summary_has_ground_truth(fallback))

            self.assertFalse(
                summary_has_ground_truth(os.path.join(d, "missing.csv"))
            )


if __name__ == "__main__":
    unittest.main()

