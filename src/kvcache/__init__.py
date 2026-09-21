"""Sharing- and session-aware request scheduling for vLLM.

This package implements a request-scheduling layer that sits *in front of*
vLLM and controls submission order / timing / prioritization. It does
not touch vLLM's internal cache or eviction. vLLM's real, built-in
automatic prefix caching runs unmodified.

Top-level components:

  - `session`, `workload`: multi-session workload generator with
    built-in cross-session content overlap
  - `prefix`: vLLM-style chained block hashes; what another session's
    prompt could let the engine reuse
  - `cachesim`: CPU model of the prefix cache for eviction-headroom studies
  - `policies`: four dispatch policies (FIFO / session-aware /
    sharing-aware / combined)
  - `vllm_backend`: thin async wrapper around `vllm.AsyncLLMEngine`
  - `bench`: benchmark harness driving workload + policy + vLLM
  - `metrics`: time-windowed metrics + incremental CSV writing
  - `analysis`: headline / ablation / fairness charts from CSV results
"""

from .session import LiveSessions, Session, Turn
from .prefix import BLOCK_SIZE, PrefixIndex, block_hashes
from .workload import Workload, WorkloadConfig, TurnEvent, generate_workload
from .policies import (
    DispatchPolicy,
    QueuedRequest,
    DispatchDecision,
    FIFOPolicy,
    SessionAwarePolicy,
    SharingAwarePolicy,
    CombinedPolicy,
)
from .vllm_backend import (
    BackendConfig,
    RequestResult,
    VLLMBackend,
    VLLMUnavailable,
    tokens_to_text,
    token_id_to_text,
)
from .metrics import MetricsConfig, MetricsLogger, RequestRecord
from .bench import BenchConfig, RunSummary, run_benchmark, MockVLLMBackend
from .analysis import RunSet, run_analysis

__all__ = [
    "LiveSessions",
    "Session",
    "Turn",
    "BLOCK_SIZE",
    "PrefixIndex",
    "block_hashes",
    "Workload",
    "WorkloadConfig",
    "TurnEvent",
    "generate_workload",
    "DispatchPolicy",
    "QueuedRequest",
    "DispatchDecision",
    "FIFOPolicy",
    "SessionAwarePolicy",
    "SharingAwarePolicy",
    "CombinedPolicy",
    "BackendConfig",
    "RequestResult",
    "VLLMBackend",
    "VLLMUnavailable",
    "tokens_to_text",
    "token_id_to_text",
    "MetricsConfig",
    "MetricsLogger",
    "RequestRecord",
    "BenchConfig",
    "RunSummary",
    "run_benchmark",
    "MockVLLMBackend",
    "RunSet",
    "run_analysis",
]

