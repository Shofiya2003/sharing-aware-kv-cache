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

from kvcache.session import LiveSessions, Session, Turn  # noqa: E402
from kvcache.prefix import BLOCK_SIZE, PrefixIndex, block_hashes  # noqa: E402
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


class TestPrefixIndex(unittest.TestCase):
    """Sharing must mean what vLLM can reuse: identical from token 0."""

    DOC = tuple(range(1000, 1000 + 3 * BLOCK_SIZE))

    def test_hashes_chain_from_token_zero(self):
        a = block_hashes(self.DOC + (1,) * BLOCK_SIZE)
        b = block_hashes(self.DOC + (2,) * BLOCK_SIZE)
        self.assertEqual(a[:3], b[:3])
        self.assertNotEqual(a[3], b[3])

    def test_same_text_mid_prompt_does_not_match(self):
        # The same block of text after a different opening hashes
        # differently -- exactly why vLLM cannot reuse it.
        a = block_hashes(self.DOC)
        b = block_hashes((7,) * BLOCK_SIZE + self.DOC)
        self.assertFalse(set(a) & set(b))

    def test_only_full_blocks_count(self):
        self.assertEqual(len(block_hashes(tuple(range(2 * BLOCK_SIZE - 1)))), 1)

    def test_shared_prefix_counts_other_sessions_only(self):
        idx = PrefixIndex()
        idx.add("s0", self.DOC + (5,) * BLOCK_SIZE)
        n, others = idx.shared_prefix("s1", self.DOC + (9,) * BLOCK_SIZE)
        self.assertEqual(n, 3 * BLOCK_SIZE)
        self.assertEqual(others, {"s0"})
        # A session's own history is not sharing.
        self.assertEqual(idx.shared_prefix("s0", self.DOC)[0], 0)

    def test_mid_prompt_doc_is_not_shared(self):
        idx = PrefixIndex()
        idx.add("s0", (1,) * BLOCK_SIZE + self.DOC)
        self.assertEqual(idx.shared_prefix("s1", (2,) * BLOCK_SIZE + self.DOC)[0], 0)

    def test_capacity_forgets_least_recent(self):
        idx = PrefixIndex(capacity_blocks=3)
        idx.add("s0", self.DOC)
        idx.add("s1", (4,) * BLOCK_SIZE)
        self.assertEqual(len(idx), 3)
        self.assertEqual(idx.cached_prefix(self.DOC), 0)  # its block 0 went first


class TestLiveSessions(unittest.TestCase):
    """Policies must only see history that has already happened."""

    def test_state_reflects_only_observed_turns(self):
        live = LiveSessions()
        live.observe("s0", 0, 1.0)
        live.observe("s0", 1, 2.0)
        s = live["s0"]
        self.assertEqual(s.turn_count, 2)
        self.assertEqual(s.last_turn_t, 2.0)
        self.assertEqual(s.total_idle_intervals, 0)

    def test_idle_gap_recorded(self):
        live = LiveSessions(idle_gap_threshold_s=5.0)
        live.observe("s0", 0, 0.0)
        live.observe("s0", 1, 30.0)
        self.assertEqual(live["s0"].total_idle_intervals, 1)
        self.assertAlmostEqual(live["s0"].avg_idle_gap(), 30.0)

    def test_out_of_order_completion_does_not_rewind(self):
        live = LiveSessions()
        live.observe("s0", 1, 10.0)
        live.observe("s0", 0, 9.0)
        self.assertEqual(live["s0"].last_turn_t, 10.0)
        self.assertEqual(live["s0"].turn_count, 2)

    def test_benchmark_policy_never_sees_future_turns(self):
        """Every time a policy scores the queue, each session's history
        must contain only turns issued at or before `now`."""
        from kvcache import bench as B

        w = generate_workload(WorkloadConfig(num_sessions=4, sim_window_s=20, seed=3,
                                             mean_idle_gap_s=3))
        seen_violation = []
        calls = []

        class Spy(SessionAwarePolicy):
            def score_queue(self, queue, sessions, prefix_index, now):
                calls.append(now)
                for sess in sessions.values():
                    if any(t.t > now + 1e-9 for t in sess.turns):
                        seen_violation.append(sess.session_id)
                return super().score_queue(queue, sessions, prefix_index, now)

        orig = B.make_policy
        B.make_policy = lambda name, alpha=0.5: Spy()
        try:
            async def go():
                with tempfile.TemporaryDirectory() as tmp:
                    backend = MockVLLMBackend(BackendConfig(max_num_seqs=2))
                    await backend.start()
                    cfg = BenchConfig(policy_name="session-aware",
                                      backend=BackendConfig(max_num_seqs=2),
                                      max_new_tokens=4, speed_factor=20.0,
                                      output_dir=tmp, run_label="spy")
                    await run_benchmark(w, cfg, backend=backend)
            asyncio.run(go())
        finally:
            B.make_policy = orig
        self.assertGreater(len(calls), 10)
        self.assertEqual(seen_violation, [])


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
        d = FIFOPolicy().score_queue(queue, LiveSessions(), PrefixIndex(), 0.0)
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

        oi = PrefixIndex()
        # Another session already sent a prompt with the sharer's opening.
        oi.add("earlier", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20))

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


class TestSharedContentAlignment(unittest.TestCase):
    """Cross-session reuse needs a common prefix from token 0.

    vLLM chains block hashes from the first token, so two sessions only share
    cache if their prompts match from position 0. These tests pin the two
    workload modes that make that testable, and the volume-matched control
    that isolates alignment from token count.
    """

    BIG = dict(num_sessions=12, sim_window_s=200, seed=0,
               shared_doc_min_tokens=800, shared_doc_max_tokens=1200,
               num_shared_docs=2, overlap_fraction=0.8,
               turn_min_tokens=16, turn_max_tokens=48,
               max_context_tokens=3900)

    @staticmethod
    def _common_prefix(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _cross_session_reuse(self, position):
        import itertools
        from collections import defaultdict
        from kvcache.workload import generate_workload, WorkloadConfig  # noqa: E402

        w = generate_workload(
            WorkloadConfig(shared_attach_position=position, **self.BIG)
        )
        by = defaultdict(list)
        for e in w.events:
            by[e.session_id].append(e)
        for v in by.values():
            v.sort(key=lambda e: e.turn_index)
        heads = [evs[-1].prompt_tokens for evs in by.values()]
        fracs = [
            self._common_prefix(p1, p2) / max(len(p1), len(p2))
            for p1, p2 in itertools.combinations(heads, 2)
        ]
        return w, fracs

    def test_session_preamble_creates_cross_session_prefix(self):
        _w, fracs = self._cross_session_reuse("session_preamble")
        self.assertGreater(max(fracs), 0.05)
        self.assertGreater(sum(1 for f in fracs if f > 0.001), 0)

    def test_session_mid_creates_none(self):
        """The control must yield zero cross-session reuse."""
        _w, fracs = self._cross_session_reuse("session_mid")
        self.assertEqual(max(fracs), 0.0)

    def test_turn_prefix_is_not_a_prompt_prefix(self):
        """'prefix' attaches to the TURN, which is mid-prompt once context
        accumulates. This is why it cannot be used as the aligned arm."""
        _w, fracs = self._cross_session_reuse("prefix")
        import statistics as st
        pre_w, pre_f = self._cross_session_reuse("session_preamble")
        self.assertLess(st.mean(fracs), st.mean(pre_f))

    def test_alignment_arms_are_volume_matched(self):
        """session_mid vs session_preamble must differ only in alignment."""
        mid, _ = self._cross_session_reuse("session_mid")
        pre, _ = self._cross_session_reuse("session_preamble")
        mid_tok = sum(len(e.prompt_tokens) for e in mid.events)
        pre_tok = sum(len(e.prompt_tokens) for e in pre.events)
        self.assertLess(abs(mid_tok - pre_tok) / max(mid_tok, pre_tok), 0.10)
        mid_tr = sum(1 for e in mid.events if e.context_truncated) / len(mid.events)
        pre_tr = sum(1 for e in pre.events if e.context_truncated) / len(pre.events)
        self.assertLess(abs(mid_tr - pre_tr), 0.05)

    def test_pinned_preamble_survives_truncation(self):
        """Truncation must drop middle turns, never the shared preamble.

        Dropping oldest-first would delete the doc at position 0 and wipe out
        cross-session reuse for exactly the long sessions that matter most.
        """
        from collections import defaultdict
        from kvcache.workload import generate_workload, WorkloadConfig  # noqa: E402

        w = generate_workload(
            WorkloadConfig(shared_attach_position="session_preamble", **self.BIG)
        )
        by = defaultdict(list)
        for e in w.events:
            by[e.session_id].append(e)
        checked = 0
        for evs in by.values():
            evs.sort(key=lambda e: e.turn_index)
            trunc = [e for e in evs if e.context_truncated]
            if not trunc:
                continue
            head = evs[0].prompt_tokens[:64]
            for e in trunc:
                self.assertEqual(e.prompt_tokens[:64], head)
                checked += 1
        self.assertGreater(checked, 0, "no truncated events to verify")


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


class _FailingBackend(MockVLLMBackend):
    """Mock whose engine "dies" after `ok` requests, or fails listed ones."""

    def __init__(self, cfg, ok=10**9, fail_at=()):
        super().__init__(cfg)
        self._ok = ok
        self._fail_at = set(fail_at)
        self._n = 0

    async def submit(self, prompt, session_id, turn_index, submit_t,
                     max_new_tokens=32, sampling_params=None):
        self._n += 1
        if self._n > self._ok or self._n in self._fail_at:
            rid = f"{session_id}__t{turn_index}__dead{self._n}"
            fut = asyncio.get_running_loop().create_future()
            fut.set_exception(RuntimeError("EngineDeadError (simulated)"))
            self._pending[rid] = fut
            return rid
        return await super().submit(prompt, session_id, turn_index, submit_t,
                                    max_new_tokens, sampling_params)


class TestEngineFailure(unittest.TestCase):
    """Round 2's fifo_generous_s2: a dead engine must not produce a 'run'."""

    def _run(self, backend, tmp):
        w = generate_workload(
            WorkloadConfig(num_sessions=4, sim_window_s=60, seed=2,
                           mean_idle_gap_s=2))
        cfg = BenchConfig(
            policy_name="fifo", backend=BackendConfig(max_num_seqs=2),
            max_new_tokens=4, output_dir=tmp, run_label="dead",
            speed_factor=300.0, discard_warmup_windows=0,
        )

        async def go():
            await backend.start()
            try:
                return await run_benchmark(w, cfg, backend=backend)
            finally:
                await backend.stop()
        return asyncio.run(go())

    def test_dead_engine_aborts_and_is_set_aside(self):
        from kvcache.bench import EngineFailure
        with tempfile.TemporaryDirectory() as tmp:
            b = _FailingBackend(BackendConfig(max_num_seqs=2), ok=8)
            with self.assertRaises(EngineFailure):
                self._run(b, tmp)
            # Not left where the launcher would count it as done ...
            self.assertFalse(os.path.exists(os.path.join(tmp, "summary_dead.csv")))
            # ... but kept for inspection.
            self.assertTrue(os.path.exists(
                os.path.join(tmp, "_aborted", "summary_dead.csv")))

    def test_failed_request_is_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = _FailingBackend(BackendConfig(max_num_seqs=2), fail_at={3})
            r = self._run(b, tmp)
            s = r.summary
            self.assertEqual(s["n_failed"], 1)
            # The failed request has no cache counter; had it been scored,
            # coverage would drop below 1 and basis would read "mixed".
            self.assertEqual(s["hit_basis"], "cached_tokens")
            self.assertEqual(s["cache_ground_truth_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()



class TestCacheSim(unittest.TestCase):
    """The CPU prefix-cache model behind notebooks/cpu_cache_headroom.ipynb."""

    @staticmethod
    def _events():
        return generate_workload(WorkloadConfig(
            num_sessions=5, sim_window_s=120, seed=2, max_context_tokens=1024,
            turn_min_tokens=16, turn_max_tokens=48)).events

    def test_repeat_prompt_reuses_all_but_last_block(self):
        from kvcache.cachesim import simulate
        from kvcache.workload import TurnEvent
        toks = tuple(range(5 * BLOCK_SIZE))
        ev = [TurnEvent("a", 0, 0.0, toks, "user", True, toks),
              TurnEvent("b", 0, 1.0, toks, "user", True, toks)]
        r = simulate(ev, None)
        # vLLM recomputes at least one token, so the last full block is not reused.
        self.assertEqual(r.cached_tokens, 4 * BLOCK_SIZE)

    def test_policies_are_bounded_by_oracle_and_ceiling(self):
        from kvcache.cachesim import POLICIES, simulate
        ev = self._events()
        ceil = simulate(ev, None).cached_token_rate
        for cap in (40, 120):
            r = {p: simulate(ev, cap, p).cached_token_rate for p in POLICIES}
            for p in POLICIES:
                self.assertLessEqual(r[p], ceil + 1e-12, (cap, p))
            self.assertGreaterEqual(r["oracle"] + 1e-12, r["lru"], cap)

    def test_lru_improves_with_capacity(self):
        from kvcache.cachesim import simulate
        ev = self._events()
        rates = [simulate(ev, c, "lru").cached_token_rate for c in (20, 60, 200, 2000)]
        self.assertEqual(rates, sorted(rates))


class TestReusePredictor(unittest.TestCase):
    """src/kvcache/predictor.py: learned from past conversations only."""

    @staticmethod
    def _conv(times, users=None):
        from kvcache.wildchat import Conversation
        c = Conversation(conv_id=str(times), start=0.0, turn_times=list(times),
                         user_text=[""] * len(times), reply_text=[""] * len(times))
        if users is not None:
            c.user_tokens = [tuple(range(n)) for n in users]
        return c

    def test_end_probability_and_gaps(self):
        from kvcache.predictor import ReturnModel
        # Two single-turn conversations end; one continues after 10 s.
        m = ReturnModel.fit([self._conv([0.0]), self._conv([0.0]), self._conv([0.0, 10.0])])
        self.assertAlmostEqual(m.p_end[0], 2 / 3)
        self.assertEqual(m.gaps[0], [10.0])

    def test_idle_time_lowers_return_probability(self):
        from kvcache.predictor import ReturnModel
        # Gaps of 10 s and 1000 s: after 60 s idle only the slow one remains.
        m = ReturnModel.fit([self._conv([0.0, 10.0]), self._conv([0.0, 1000.0])])
        self.assertAlmostEqual(m.p_return_within(1, 0.0, 60.0), 0.5)
        self.assertAlmostEqual(m.p_return_within(1, 60.0, 60.0), 0.0)
        self.assertAlmostEqual(m.p_return_within(1, 60.0, 1000.0), 1.0)

    def test_conversations_that_always_end_never_return(self):
        from kvcache.predictor import ReturnModel
        m = ReturnModel.fit([self._conv([0.0])] * 5)
        self.assertEqual(m.p_return_within(1, 0.0, 1e9), 0.0)

    def test_fit_probability(self):
        from kvcache.predictor import FitModel
        f = FitModel.fit([self._conv([0.0, 1.0, 2.0], users=[5, 100, 300])], max_context=1000)
        self.assertEqual(f.p_fits(500), 1.0)      # room 500: both follow-ups fit
        self.assertEqual(f.p_fits(800), 0.5)      # room 200: only the 100-token one
        self.assertEqual(f.p_fits(1000), 0.0)     # no room


class TestCacheSimWildChatPolicies(unittest.TestCase):
    """Reply caching and the lfu / preble-cost / reuse policies."""

    def test_reply_blocks_are_cached_for_the_next_turn(self):
        from kvcache.cachesim import simulate
        from kvcache.workload import TurnEvent
        p1 = tuple(range(2 * BLOCK_SIZE))
        reply = tuple(range(500, 500 + 2 * BLOCK_SIZE))
        p2 = p1 + reply + tuple(range(900, 900 + BLOCK_SIZE))
        ev = [TurnEvent("a", 0, 0.0, p1, "user", True, p1, output_tokens=reply),
              TurnEvent("a", 1, 5.0, p2, "user", True, p2)]
        # Turn 2 reuses turn 1's prompt AND reply: 4 blocks.
        self.assertEqual(simulate(ev, None).cached_tokens, 4 * BLOCK_SIZE)

    def test_new_policies_are_bounded(self):
        from kvcache.cachesim import ALL_POLICIES, simulate
        from kvcache.predictor import FitModel, ReturnModel, ReusePredictor
        from kvcache.wildchat import Conversation
        ev = TestCacheSim._events()
        # A predictor fitted on a toy history is enough to exercise the policy.
        hist = [Conversation("h", 0.0, [0.0, 20.0, 50.0], [""] * 3, [""] * 3,
                             user_tokens=[(1,) * 30] * 3, reply_tokens=[()] * 3)]
        pred = ReusePredictor(ReturnModel.fit(hist), FitModel.fit(hist, 4096))
        ceil = simulate(ev, None).cached_token_rate
        r = {p: simulate(ev, 60, p, predictor=pred).cached_token_rate for p in ALL_POLICIES}
        for p, v in r.items():
            self.assertLessEqual(v, ceil + 1e-12, p)
            self.assertGreaterEqual(r["oracle"] + 1e-12, v, p)

    def test_reuse_needs_a_predictor(self):
        from kvcache.cachesim import simulate
        with self.assertRaises(ValueError):
            simulate(TestCacheSim._events(), 60, "reuse")
