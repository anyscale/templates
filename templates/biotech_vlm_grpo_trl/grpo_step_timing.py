"""
grpo_step_timing.py  --  phase-level observability for TRL GRPOTrainer.

Two-line install, no changes to the training script otherwise:

    from grpo_step_timing import instrument_grpo_trainer
    trainer = GRPOTrainer(...)
    wrapped = instrument_grpo_trainer(trainer)   # see options below
    trainer.train()

Every optimizer step emits a flat dict of metrics into the HF log stream (so W&B /
Trackio / TensorBoard pick them up with zero config, and they land in
`trainer.state.log_history`) and, when running inside a Ray Train worker, also via
`ray.train.report` so a Ray Train `UserCallback` on the driver can consume them.

Metric families (all seconds unless suffixed):

  timing/step_s                     whole optimizer step (on_step_begin -> on_step_end)
  timing/dataloader_s               step begin -> first generate call
  timing/tokenize_s                 prompt tokenization + image preprocessing (CPU, processor)
  timing/generate_s                 rollout wall clock (HF generate or vLLM call), THIS rank only
  timing/generate/prefill_s         HF generate only: entry -> first decode token (incl. ViT forward)
  timing/generate/decode_s          HF generate only: first -> last decode token
  timing/generate/ms_per_token      decode_s / decode steps
  timing/sync_wait_s                dist.barrier() right after this rank's generate returns:
                                    how long the fast ranks waited for the slowest rollout
  timing/reward_s                   all reward funcs (+ TRL's cross-rank gather of rewards)
  timing/reward/<fn_name>_s         each reward func separately
  timing/forward_s                  compute_loss, summed over micro-batches
  timing/backward_s                 accelerator.backward, summed over micro-batches
                                    (DDP gradient all-reduce lives in here)
  timing/optimizer_s                optimizer.step
  timing/other_s                    step_s - everything above (decode, padding, advantages, ...)
  timing/*_frac                     phase / step_s for generate, reward, forward, backward, sync_wait

  rollout/completion_len_{min,max,mean}       tokens (this rank)
  rollout/padding_waste_frac                  sum(max_len - len_i) / (N * max_len)   [HF batched generate
                                              runs every sequence to the longest; this is the idle share]
  rollout/tokens_per_s                        completion tokens / generate_s   (this rank)
  rollout/samples_per_gpu_hour                completions / (step_s / 3600)     (this rank, per GPU)
  rollout/generate_s_{rank_min,rank_max,rank_spread}   all_gather across ranks; spread = the barrier
                                                        time the fastest rank paid

  gpu/util_pct/<phase>                        mean SM utilization sampled every 100 ms, tagged by phase
  gpu/mem_peak_gb/<phase>                     torch.cuda.max_memory_allocated per phase

  profiler/trace_path                         when a torch.profiler capture ran this step

Options for instrument_grpo_trainer(trainer, ...):
  gpu_sampler=True            background pynvml sampler (needs `pip install nvidia-ml-py`)
  barrier_after_generate=True adds dist.barrier() after generate to measure straggler wait;
                              slight perturbation, only active in train mode
  profile_every=0             >0: torch.profiler capture of one full step every N steps
  profile_dir="./traces"      where Chrome traces land (Perfetto-viewable)
  report_to_ray=True          ray.train.report from every rank when inside a Ray Train worker
                              (Ray Train V2 makes report() a barrier, so all ranks must call it)

Method names target TRL 1.13 (verified against the installed source). Wrapping is
best-effort: a missing method skips that phase rather than crashing the run;
`instrument_grpo_trainer` returns a dict of what was actually wrapped so you can
assert on it in a smoke test. `optimizer` flips to True lazily, on the first step,
because the Trainer only builds the optimizer inside train().

Changes relative to the original draft of this file (the draft was truncated mid
`on_step_end`; the tail was rewritten and these three things fixed):

  1. "generate" now wraps `_generate_single_turn` (the pure HF/vLLM rollout) instead
     of `_generate`. `_generate` also tokenizes and runs `accelerator.gather` on the
     completion lengths, which is a cross-rank sync -- wrapping it would have hidden
     the straggler wait inside generate_s and made `generate_s_rank_spread` ~0.
  2. The sync-wait barrier moved from `on_optimizer_step` to right after
     `_generate_single_turn`. `on_optimizer_step` fires *after* optimizer.step in
     transformers 5.x, and DDP's all-reduce has already synchronized the ranks in
     backward by then, so a barrier there measures nothing. The first cross-rank
     sync of a GRPO step is the gather inside `_generate` / `_calculate_rewards`;
     the barrier now sits right before it, so `sync_wait_s` is exactly the time a
     fast rank spends waiting for the slowest rank's rollout.
  3. Metrics are injected by wrapping `trainer.log` rather than from an `on_log`
     callback: integration callbacks (TensorBoard, W&B) are registered before any
     callback we add, so an `on_log` mutation would arrive after they already logged.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from collections import defaultdict
from typing import Any, Callable

import torch
import torch.distributed as dist
from transformers import LogitsProcessor, LogitsProcessorList, TrainerCallback

# --------------------------------------------------------------------------------------
# optional deps
# --------------------------------------------------------------------------------------
try:
    import ray.train as _ray_train

    def _ray_rank() -> int | None:
        try:
            return _ray_train.get_context().get_world_rank()
        except Exception:
            return None

except ImportError:

    def _ray_rank() -> int | None:
        return None


try:
    import pynvml

    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


def _dist_ok() -> bool:
    return dist.is_available() and dist.is_initialized()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------------------
class _StepState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t: dict[str, float] = defaultdict(float)  # phase -> seconds
        self.n: dict[str, int] = defaultdict(int)  # phase -> call count
        self.mem_peak: dict[str, float] = {}  # phase -> GB
        self.completion_lens: list[int] = []
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.decode_steps = 0
        self.step_start: float | None = None
        self.first_generate_at: float | None = None
        self.current_phase: str | None = None
        self.trace_path: str | None = None


# --------------------------------------------------------------------------------------
# phase timer (also drives the GPU sampler tag + peak-memory bookkeeping)
# --------------------------------------------------------------------------------------
class _PhaseTimer:
    def __init__(self, state: _StepState) -> None:
        self.state = state

    def wrap(
        self,
        obj: Any,
        attr: str,
        phase: str,
        on_enter: Callable | None = None,
        on_exit: Callable | None = None,
    ) -> bool:
        fn = getattr(obj, attr, None)
        if fn is None or not callable(fn):
            return False

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            st = self.state
            if on_enter:
                on_enter()
            prev = st.current_phase
            st.current_phase = phase
            _sync()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _sync()
                st.t[phase] += time.perf_counter() - t0
                st.n[phase] += 1
                if torch.cuda.is_available():
                    gb = torch.cuda.max_memory_allocated() / 1e9
                    st.mem_peak[phase] = max(st.mem_peak.get(phase, 0.0), gb)
                st.current_phase = prev
                if on_exit:
                    on_exit()

        setattr(obj, attr, wrapper)
        return True


# --------------------------------------------------------------------------------------
# prefill / decode split for HF generate
# --------------------------------------------------------------------------------------
class _DecodeClock(LogitsProcessor):
    """Called once per decode step by HF generate. First call marks end of prefill."""

    def __init__(self, state: _StepState) -> None:
        self.state = state
        self.t_enter: float | None = None
        self.t_first: float | None = None
        self.t_last: float | None = None
        self.steps = 0

    def arm(self) -> None:
        self.t_enter = time.perf_counter()
        self.t_first = self.t_last = None
        self.steps = 0

    def __call__(self, input_ids, scores):
        now = time.perf_counter()
        if self.t_first is None:
            self.t_first = now
        self.t_last = now
        self.steps += 1
        return scores

    def flush(self) -> None:
        if self.t_enter is None or self.t_first is None:
            return
        st = self.state
        st.prefill_s += self.t_first - self.t_enter
        st.decode_s += (self.t_last or self.t_first) - self.t_first
        st.decode_steps += self.steps


def _patch_generate_for_decode_clock(trainer, clock: _DecodeClock) -> bool:
    """
    Inject the clock into the model.generate call TRL makes. Only meaningful when
    use_vllm=False.

    TRL 1.x generates on `accelerator.unwrap_model(self.model_wrapped)`. For DDP that
    is `model_wrapped.module`, which is the very object held in `trainer.model` (the
    PEFT model when peft_config is set, else the bare HF model). So patching the
    `generate` *instance attribute* on `trainer.model` is enough: PEFT's generate
    forwards **kwargs (our `logits_processor`) down to transformers' generate.
    Not valid for DeepSpeed ZeRO-3 / FSDP, where the unwrapped object differs.
    """
    model = getattr(trainer, "model", None)
    gen = getattr(model, "generate", None)
    if gen is None:
        return False

    @functools.wraps(gen)
    def generate_wrapper(*args, **kwargs):
        lp = kwargs.get("logits_processor")
        lp = LogitsProcessorList(list(lp) if lp else [])
        lp.append(clock)
        kwargs["logits_processor"] = lp
        clock.arm()
        try:
            return gen(*args, **kwargs)
        finally:
            clock.flush()

    model.generate = generate_wrapper
    return True


# --------------------------------------------------------------------------------------
# GPU utilization sampler, tagged by current phase
# --------------------------------------------------------------------------------------
class _GpuSampler(threading.Thread):
    def __init__(self, state: _StepState, interval_s: float = 0.1) -> None:
        super().__init__(daemon=True, name="grpo-gpu-sampler")
        self.state = state
        self.interval = interval_s
        self._stop = threading.Event()
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self.ok = False
        if _HAS_NVML and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                idx = torch.cuda.current_device()
                # honor CUDA_VISIBLE_DEVICES remapping (Ray Train sets it per node)
                vis = os.environ.get("CUDA_VISIBLE_DEVICES")
                if vis:
                    idx = int(vis.split(",")[idx])
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self.ok = True
            except Exception:
                self.ok = False

    def run(self) -> None:
        if not self.ok:
            return
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                phase = self.state.current_phase or "idle"
                with self._lock:
                    self.samples[phase].append(float(util))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def drain(self) -> dict[str, float]:
        with self._lock:
            out = {k: (sum(v) / len(v)) for k, v in self.samples.items() if v}
            self.samples.clear()
        return out

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------------------
# reward wrapping (per function)
# --------------------------------------------------------------------------------------
def _wrap_reward_funcs(trainer, timer: _PhaseTimer) -> list[str]:
    funcs = getattr(trainer, "reward_funcs", None)
    if not funcs:
        return []
    names: list[str] = []
    wrapped = []
    for i, fn in enumerate(funcs):
        name = getattr(fn, "__name__", None) or getattr(getattr(fn, "config", None), "_name_or_path", None) or f"reward_{i}"
        name = str(name).split("/")[-1]
        names.append(name)
        if callable(fn) and not isinstance(fn, torch.nn.Module):
            state = timer.state

            def make(fn=fn, name=name):
                @functools.wraps(fn)
                def w(*a, **k):
                    _sync()
                    t0 = time.perf_counter()
                    try:
                        return fn(*a, **k)
                    finally:
                        _sync()
                        state.t[f"reward/{name}"] += time.perf_counter() - t0

                return w

            wrapped.append(make())
        else:
            # nn.Module reward models: TRL calls them internally; time via forward hook
            state = timer.state

            def pre(mod, inp, name=name):
                _sync()
                mod.__grpo_t0 = time.perf_counter()

            def post(mod, inp, out, name=name):
                _sync()
                state.t[f"reward/{name}"] += time.perf_counter() - getattr(mod, "__grpo_t0", time.perf_counter())

            fn.register_forward_pre_hook(pre)
            fn.register_forward_hook(post)
            wrapped.append(fn)
    trainer.reward_funcs = wrapped
    return names


# --------------------------------------------------------------------------------------
# completion-length capture: hook the reward stage, which receives completion_ids
# --------------------------------------------------------------------------------------
def _wrap_completion_capture(trainer, state: _StepState) -> bool:
    fn = getattr(trainer, "_calculate_rewards", None)
    if fn is None:
        return False

    @functools.wraps(fn)
    def w(*args, **kwargs):
        # TRL 1.x signature: (inputs, prompts, completions, completion_ids_list)
        cids = kwargs.get("completion_ids_list")
        if cids is None and len(args) >= 4:
            cids = args[3]
        if cids is not None:
            try:
                state.completion_lens.extend(len(c) for c in cids)
            except Exception:
                pass
        return fn(*args, **kwargs)

    trainer._calculate_rewards = w
    return True


# --------------------------------------------------------------------------------------
# main callback: assemble + emit
# --------------------------------------------------------------------------------------
class GRPOStepTimingCallback(TrainerCallback):
    def __init__(
        self,
        trainer,
        state: _StepState,
        timer: _PhaseTimer,
        sampler: _GpuSampler | None,
        profile_every: int,
        profile_dir: str,
        report_to_ray: bool,
        wrapped: dict[str, Any],
    ) -> None:
        self.trainer = trainer
        self.state = state
        self.timer = timer
        self.sampler = sampler
        self.profile_every = profile_every
        self.profile_dir = profile_dir
        self.report_to_ray = report_to_ray
        self.wrapped = wrapped
        self.pending: dict[str, Any] | None = None
        self._prof = None

    # ---- lifecycle ---------------------------------------------------------------
    def on_train_begin(self, args, state, control, **kwargs):
        if self.sampler and self.sampler.ok:
            self.sampler.start()

    def on_train_end(self, args, state, control, **kwargs):
        if self.sampler:
            self.sampler.stop()

    def on_step_begin(self, args, state, control, **kwargs):
        st = self.state
        st.reset()
        st.step_start = time.perf_counter()
        st.current_phase = "dataloader"
        # optimizer only exists once train() has started; wrap it on the first step
        if not self.wrapped.get("optimizer"):
            opt = getattr(self.trainer, "optimizer", None)
            if opt is not None:
                self.wrapped["optimizer"] = self.timer.wrap(opt, "step", "optimizer")
        if self.profile_every and (state.global_step + 1) % self.profile_every == 0:
            os.makedirs(self.profile_dir, exist_ok=True)
            self._prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            )
            self._prof.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        st = self.state
        if st.step_start is None:
            return
        _sync()
        total = time.perf_counter() - st.step_start
        st.current_phase = None

        if self._prof is not None:
            self._prof.__exit__(None, None, None)
            rank = _ray_rank()
            if rank is None:
                rank = dist.get_rank() if _dist_ok() else 0
            path = os.path.join(self.profile_dir, f"grpo_step{state.global_step}_rank{rank}.json")
            try:
                self._prof.export_chrome_trace(path)
                st.trace_path = path
            except Exception:
                pass
            self._prof = None

        t = st.t
        m: dict[str, Any] = {
            "timing/step_s": total,
            "timing/dataloader_s": (st.first_generate_at - st.step_start) if st.first_generate_at else 0.0,
            "timing/tokenize_s": t.get("tokenize", 0.0),
            "timing/generate_s": t.get("generate", 0.0),
            "timing/sync_wait_s": t.get("sync_wait", 0.0),
            "timing/reward_s": t.get("reward", 0.0),
            "timing/forward_s": t.get("forward", 0.0),
            "timing/backward_s": t.get("backward", 0.0),
            "timing/optimizer_s": t.get("optimizer", 0.0),
        }
        if st.decode_steps:
            m["timing/generate/prefill_s"] = st.prefill_s
            m["timing/generate/decode_s"] = st.decode_s
            m["timing/generate/ms_per_token"] = 1000.0 * st.decode_s / max(st.decode_steps, 1)
        for k, v in t.items():
            if k.startswith("reward/"):
                m[f"timing/{k}_s"] = v

        known = sum(
            m[k]
            for k in (
                "timing/dataloader_s",
                "timing/tokenize_s",
                "timing/generate_s",
                "timing/sync_wait_s",
                "timing/reward_s",
                "timing/forward_s",
                "timing/backward_s",
                "timing/optimizer_s",
            )
        )
        m["timing/other_s"] = max(total - known, 0.0)
        if total > 0:
            for k in ("generate", "reward", "forward", "backward", "sync_wait"):
                m[f"timing/{k}_frac"] = m[f"timing/{k}_s"] / total

        # rollout shape
        lens = st.completion_lens
        if lens:
            mx = max(lens)
            m["rollout/completion_len_min"] = min(lens)
            m["rollout/completion_len_max"] = mx
            m["rollout/completion_len_mean"] = sum(lens) / len(lens)
            m["rollout/padding_waste_frac"] = sum(mx - l for l in lens) / (len(lens) * mx) if mx else 0.0
            if m["timing/generate_s"] > 0:
                m["rollout/tokens_per_s"] = sum(lens) / m["timing/generate_s"]
            m["rollout/samples_per_gpu_hour"] = len(lens) / (total / 3600.0)

        # straggler spread across ranks (all ranks reach on_step_end, so all_gather is safe)
        if _dist_ok():
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            g = torch.tensor([m["timing/generate_s"]], device=dev)
            gathered = [torch.zeros_like(g) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, g)
            vals = [x.item() for x in gathered]
            m["rollout/generate_s_rank_min"] = min(vals)
            m["rollout/generate_s_rank_max"] = max(vals)
            m["rollout/generate_s_rank_spread"] = max(vals) - min(vals)

        # GPU utilization per phase + peak memory per phase
        if self.sampler is not None and self.sampler.ok:
            for phase, util in self.sampler.drain().items():
                m[f"gpu/util_pct/{phase}"] = util
        for phase, gb in st.mem_peak.items():
            m[f"gpu/mem_peak_gb/{phase}"] = gb

        if st.trace_path:
            m["profiler/trace_path"] = st.trace_path

        self.pending = m

    # ---- emission -------------------------------------------------------------------
    def inject_into_logs(self, logs: dict) -> None:
        """Called from the wrapped trainer.log on every rank, for train-mode logs only."""
        if self.pending is None:
            return
        logs.update(self.pending)
        if self.report_to_ray and _ray_rank() is not None:
            # Ray Train V2: report() is a barrier, so every rank reports (each with its
            # own rank-local numbers). Only rank 0's dict is attached to checkpoints;
            # a driver-side UserCallback sees all of them.
            numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float, str))}
            try:
                _ray_train.report(numeric)
            except Exception:
                pass
        self.pending = None


def _wrap_trainer_log(trainer, cb: GRPOStepTimingCallback) -> bool:
    fn = getattr(trainer, "log", None)
    if fn is None:
        return False

    @functools.wraps(fn)
    def log(logs, *args, **kwargs):
        if isinstance(logs, dict) and getattr(trainer.model, "training", True):
            cb.inject_into_logs(logs)
        return fn(logs, *args, **kwargs)

    trainer.log = log
    return True


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------
def instrument_grpo_trainer(
    trainer,
    gpu_sampler: bool = True,
    barrier_after_generate: bool = True,
    profile_every: int = 0,
    profile_dir: str = "./traces",
    report_to_ray: bool = True,
) -> dict[str, Any]:
    state = _StepState()
    timer = _PhaseTimer(state)
    wrapped: dict[str, Any] = {}

    def _mark_first_generate():
        if state.first_generate_at is None:
            state.first_generate_at = time.perf_counter()

    def _straggler_barrier():
        # Only in train mode: eval batches are not guaranteed to be symmetric across ranks.
        if barrier_after_generate and _dist_ok() and getattr(trainer.model, "training", False):
            _sync()
            t0 = time.perf_counter()
            dist.barrier()
            state.t["sync_wait"] += time.perf_counter() - t0

    wrapped["tokenize"] = timer.wrap(trainer, "_tokenize_prompts", "tokenize")
    wrapped["generate"] = timer.wrap(
        trainer, "_generate_single_turn", "generate", on_enter=_mark_first_generate, on_exit=_straggler_barrier
    )
    # order matters: capture first (inner), then time (outer) so completion capture is inside "reward"
    wrapped["completion_capture"] = _wrap_completion_capture(trainer, state)
    wrapped["reward_total"] = timer.wrap(trainer, "_calculate_rewards", "reward")
    wrapped["reward_funcs"] = _wrap_reward_funcs(trainer, timer)
    wrapped["forward"] = timer.wrap(trainer, "compute_loss", "forward")
    wrapped["backward"] = timer.wrap(trainer.accelerator, "backward", "backward") if getattr(trainer, "accelerator", None) else False
    wrapped["optimizer"] = False  # wrapped lazily on the first on_step_begin

    clock = _DecodeClock(state)
    wrapped["prefill_decode_split"] = (not getattr(trainer, "use_vllm", False)) and _patch_generate_for_decode_clock(trainer, clock)

    sampler = _GpuSampler(state) if gpu_sampler else None
    wrapped["gpu_sampler"] = bool(sampler and sampler.ok)

    cb = GRPOStepTimingCallback(
        trainer=trainer,
        state=state,
        timer=timer,
        sampler=sampler,
        profile_every=profile_every,
        profile_dir=profile_dir,
        report_to_ray=report_to_ray,
        wrapped=wrapped,
    )
    trainer.add_callback(cb)
    wrapped["log_hook"] = _wrap_trainer_log(trainer, cb)
    wrapped["ray_report"] = bool(report_to_ray and _ray_rank() is not None)
    return wrapped
