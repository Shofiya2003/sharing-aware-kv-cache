"""Real vLLM backend wrapper for the scheduling experiments.

This module is the *only* place in the codebase that imports `vllm`. It
exposes a tiny async API:

    backend = VLLMBackend(model="Qwen/Qwen2.5-1.5B-Instruct", gpu_memory_utilization=0.5)
    await backend.start()
    request_id = await backend.submit(prompt="...", max_new_tokens=32)
    result = await backend.wait(request_id)
    await backend.stop()

We deliberately do NOT modify vLLM's internal cache/eviction. The
scheduling layer in `policies.py` and the driver in `bench.py` decide
*what order and timing* requests are submitted; vLLM's real prefix
caching and block eviction then run as they would in production under
whatever memory pressure we configured.

`gpu_memory_utilization` is the lever we use to force real, observable
eviction pressure (see Phase 1 of the build spec). Lower it (e.g. 0.3)
to push vLLM into a constrained regime.

If vLLM is not importable (e.g. on a CPU-only dev box), the backend
raises a clear `VLLMUnavailable` error. The CLI in `run_experiment.py`
treats that as a soft skip — but `experiments/phase1_smoke.py` will
fail loudly to remind you to run the real thing on Kaggle/Colab.
"""

from __future__ import annotations

import asyncio

class VLLMUnavailable(RuntimeError):
    """Raised when vLLM cannot be imported. The benchmark requires it."""


# ---------------------------------------------------------------------------
# Token id -> text rendering
# ---------------------------------------------------------------------------
#
# Our workload generator emits integer token ids. To turn them into
# text that we can submit to a real LLM, we need a real tokenizer.
# We use the model's own tokenizer on the GPU host (loaded once at
# backend start). Each token id maps to a short ASCII string so that
# the resulting text is meaningful to the model.
#
# The mapping is deterministic and human-readable, so the same workload
# produces the same prompt on any machine.


_ALPHABET = (
    "the of and to in a is that for on with as it was by an be this are not "
    "from at or have but his they she which we one all there their what when "
    "your can said about would been if more her than them no time only do "
    "could so my some these other into make them then like over also our who "
    "very long way years use work first well water than ever little place "
    "after thing just great world life still find here something take why "
    "help put different again kind hand high mean keep never much"
).split()


def token_id_to_text(tok: int, vocab_size: int) -> str:
    """Map a token id in [0, vocab_size) to a deterministic short word.

    We do NOT use the LLM's tokenizer here (we don't have one at workload
    generation time, before the model is loaded). Instead, we map each id
    to a word from `_ALPHABET` so that the resulting prompt is composed
    of real English words and is meaningful enough for the model to
    process. The model will see prompts like:
        "the of and to in a is that for on with as it was by ..."
    which is fine for prefix-cache reuse experiments.
    """
    if vocab_size <= 0:
        return ""
    idx = tok % len(_ALPHABET)
    return _ALPHABET[idx]


def tokens_to_text(tokens, vocab_size: int) -> str:
    return " ".join(token_id_to_text(int(t), vocab_size) for t in tokens)

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BackendConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    gpu_memory_utilization: float = 0.5
    max_model_len: int = 4096
    dtype: str = "auto"
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    max_num_seqs: int = 8
    seed: int = 0


@dataclass
class RequestResult:
    request_id: str
    text: str
    submit_t: float  # simulation time at which the user requested this turn
    submit_wall_t: float  # wall-clock monotonic time at which we actually called submit
    first_token_t: float
    complete_t: float
    n_output_tokens: int
    n_prompt_tokens: int
    error: Optional[str] = None

    @property
    def latency_ms(self) -> float:
        """Wall-clock latency in milliseconds (end-to-end)."""
        if self.error:
            return 0.0
        return (self.complete_t - self.submit_wall_t) * 1000.0


class VLLMBackend:
    """Thin async wrapper around `vllm.AsyncLLMEngine`."""

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self.engine = None
        self._started = False
        self._pending: Dict[str, asyncio.Future] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        print(f"[vllm] starting engine: model={self.cfg.model} "
              f"gpu_mem={self.cfg.gpu_memory_utilization} "
              f"max_len={self.cfg.max_model_len} max_seqs={self.cfg.max_num_seqs} "
              f"prefix_caching={self.cfg.enable_prefix_caching} "
              f"enforce_eager={self.cfg.enforce_eager}")
        import inspect
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
        except Exception as e:  # noqa: BLE001
            raise VLLMUnavailable(
                f"vllm is not importable: {e!r}. Run this on a GPU host."
            ) from e

        # Only pass arguments that this vLLM version's AsyncEngineArgs
        # actually supports. Newer vLLM removed `disable_log_stats` (refactored
        # into the logging plugin system); older versions need it to be set
        # explicitly. This keeps the code portable across vLLM 0.6.x..0.10.x.
        kwargs = dict(
            model=self.cfg.model,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            max_model_len=self.cfg.max_model_len,
            dtype=self.cfg.dtype,
            enable_prefix_caching=self.cfg.enable_prefix_caching,
            enforce_eager=self.cfg.enforce_eager,
            max_num_seqs=self.cfg.max_num_seqs,
            seed=self.cfg.seed,
        )
        sig = inspect.signature(AsyncEngineArgs)
        if "disable_log_stats" in sig.parameters:
            kwargs["disable_log_stats"] = False

        args = AsyncEngineArgs(**kwargs)
        try:
            self.engine = AsyncLLMEngine.from_engine_args(args)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            hint = ""
            if "memory" in msg.lower() or "cuda" in msg.lower() or "oom" in msg.lower():
                hint = (" HINT: vLLM could not allocate with "
                        f"gpu_memory_utilization={self.cfg.gpu_memory_utilization}. "
                        "On a 16GB T4 try --gpu-memory 0.4-0.5 for Phase 1, "
                        "lower --max-model-len (e.g. 2048), or pass enforce_eager=True.")
            raise VLLMUnavailable(f"vLLM engine failed to start: {e!r}.{hint}") from e
        self._started = True
        print(f"[vllm] engine started OK: model={self.cfg.model}")
        logger.info(
            "vLLM engine started: model=%s gpu_mem=%.2f max_len=%d prefix_caching=%s",
            self.cfg.model,
            self.cfg.gpu_memory_utilization,
            self.cfg.max_model_len,
            self.cfg.enable_prefix_caching,
        )

    async def stop(self) -> None:
        for t in list(self._tasks.values()):
            t.cancel()
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._tasks.clear()
        self.engine = None
        self._started = False


    async def submit(
        self,
        prompt: str,
        session_id: str,
        turn_index: int,
        submit_t: float,
        max_new_tokens: int = 32,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Enqueue a request. Returns a request id; result via `wait`."""
        if not self._started:
            raise RuntimeError("VLLMBackend.start() must be called first")
        from vllm import SamplingParams  # type: ignore

        sp = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            **(sampling_params or {}),
        )
        request_id = f"{session_id}__t{turn_index}__{uuid.uuid4().hex[:6]}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[request_id] = fut

        async def _run():
            submit_wall = time.monotonic()
            first_token_wall = None
            text_chunks: List[str] = []
            n_out = 0
            n_prompt = 0
            try:
                async for output in self.engine.generate(  # type: ignore[attr-defined]
                    prompt, sp, request_id=request_id
                ):
                    if first_token_wall is None and output.outputs:
                        first_token_wall = time.monotonic()
                    if output.outputs:
                        text_chunks.append(output.outputs[0].text)
                        n_out = len(output.outputs[0].token_ids)
                    n_prompt = (
                        len(output.prompt_token_ids)
                        if output.prompt_token_ids is not None
                        else n_prompt
                    )
                    if output.finished:
                        break
                complete_wall = time.monotonic()
                result = RequestResult(
                    request_id=request_id,
                    text="".join(text_chunks),
                    submit_t=submit_t,
                    submit_wall_t=submit_wall,
                    first_token_t=first_token_wall or complete_wall,
                    complete_t=complete_wall,
                    n_output_tokens=n_out,
                    n_prompt_tokens=n_prompt,
                )
                if not fut.done():
                    fut.set_result(result)
            except Exception as e:  # noqa: BLE001
                if not fut.done():
                    fut.set_exception(e)

        self._tasks[request_id] = asyncio.create_task(_run())
        return request_id


    async def wait(self, request_id: str, timeout: Optional[float] = None) -> RequestResult:
        fut = self._pending[request_id]
        if timeout is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout)

    async def submit_and_wait(
        self,
        prompt: str,
        session_id: str,
        turn_index: int,
        submit_t: float,
        max_new_tokens: int = 32,
        timeout: Optional[float] = None,
    ) -> RequestResult:
        rid = await self.submit(prompt, session_id, turn_index, submit_t, max_new_tokens)
        return await self.wait(rid, timeout=timeout)

    @property
    def is_started(self) -> bool:
        return self._started

    def num_in_flight(self) -> int:
        return sum(0 if t.done() else 1 for t in self._tasks.values())

