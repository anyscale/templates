"""Exit codes the daemon writes to ``.exitcode`` when Ray fails around a task and the wrapper
never ran. All but EXIT_FRAMEWORK fall in nf-core's retry band, (130..145) + 104 + (175..177).
"""

from __future__ import annotations

EXIT_OOM = 137

EXIT_CANCELLED = 143

EXIT_NODE_LOST = 175

# Separate from 175: an input vanished, not this task's host.
EXIT_OBJECT_LOST = 176

EXIT_RUNTIME_ENV = 177

# Outside the retry band: an unrecognized framework error will not fix itself on retry.
EXIT_FRAMEWORK = 174

# Keyed by class name so this table imports and tests without Ray.
RAY_ERROR_EXIT_CODES: dict[str, int] = {
    "OutOfMemoryError": EXIT_OOM,
    "TaskCancelledError": EXIT_CANCELLED,
    "NodeDiedError": EXIT_NODE_LOST,
    "WorkerCrashedError": EXIT_NODE_LOST,
    "LocalRayletDiedError": EXIT_NODE_LOST,
    "OwnerDiedError": EXIT_NODE_LOST,
    "ActorDiedError": EXIT_NODE_LOST,
    "ObjectLostError": EXIT_OBJECT_LOST,
    "ObjectFetchTimedOutError": EXIT_OBJECT_LOST,
    "ObjectReconstructionFailedError": EXIT_OBJECT_LOST,
    "ReferenceCountingAssertionError": EXIT_OBJECT_LOST,
    "RuntimeEnvSetupError": EXIT_RUNTIME_ENV,
}

NFCORE_RETRYABLE: frozenset[int] = frozenset(
    list(range(130, 146)) + [104] + list(range(175, 178))
)


def exit_code_for(exc: BaseException) -> int:
    """The exit code for a Ray-side failure, matched along the MRO so new Ray subclasses map."""
    for klass in type(exc).__mro__:
        code = RAY_ERROR_EXIT_CODES.get(klass.__name__)
        if code is not None:
            return code
    return EXIT_FRAMEWORK


def describe(exc: BaseException) -> str:
    """A one-line summary for the ``.nf-ray.log`` beside the task."""
    code = exit_code_for(exc)
    retry = "retryable" if code in NFCORE_RETRYABLE else "not retryable"
    text = str(exc).strip().splitlines()
    head = text[0] if text else ""
    return f"{type(exc).__name__}: {head} -> .exitcode {code} ({retry} under nf-core base.config)"


def is_retryable(exc: BaseException) -> bool:
    """Whether nf-core's default ``errorStrategy`` would retry this failure."""
    return exit_code_for(exc) in NFCORE_RETRYABLE
