"""Map a Ray failure onto an exit code Nextflow's ``errorStrategy`` understands.

Nextflow decides whether to retry a task by looking at ``.exitcode`` in the task's
work directory. Normally the job wrapper writes that file itself, on its way out.
But when Ray fails *around* the task -- the node was reclaimed, the object store
lost an input, the memory monitor killed the worker -- the wrapper never runs and
never gets to write anything. Left alone, Nextflow waits ``exitReadTimeout`` and
then reports an opaque "failed to get exit status".

So :mod:`nf_ray.daemon` writes ``.exitcode`` on Ray's behalf in exactly that case,
using the codes below. They are chosen to land inside the band nf-core's
``conf/base.config`` already treats as retryable::

    errorStrategy = { task.exitStatus in ((130..145) + 104 + (175..177)) ? 'retry' : 'finish' }

which means an unmodified nf-core pipeline gets sensible behaviour from a Ray
cluster with no edits at all. Two of the mappings do real work:

* **OOM becomes 137.** Ray's memory monitor killing a worker is the same event as
  the kernel's OOM killer, and 137 is what nf-core's retry logic expects for it.
  Because nf-core scales every request by ``task.attempt``, the retry comes back
  asking for twice the memory -- which is the correct response and happens with
  no intervention.
* **Preemption becomes 175, not a Nextflow-level failure.** A spot instance going
  away is infrastructure, not a bad task, and conflating the two is how a
  pipeline burns its ``maxRetries`` budget on events that were never its fault.

A code *outside* the retryable band is deliberate for the cases that will not fix
themselves on a second attempt.
"""

from __future__ import annotations

#: Ray's memory monitor killed the worker. SIGKILL convention; nf-core retries
#: and, because requests scale by `task.attempt`, asks for more memory.
EXIT_OOM = 137

#: The task was cancelled (``ray.cancel``). SIGTERM convention.
EXIT_CANCELLED = 143

#: The node, worker, raylet or owner died -- the preemption class of failure.
EXIT_NODE_LOST = 175

#: An object was lost or could not be fetched. Distinct from 175 because it
#: usually means an *input* vanished rather than this task's host dying, and the
#: two want different diagnoses even though both retry.
EXIT_OBJECT_LOST = 176

#: ``runtime_env`` setup failed -- most often an image that could not be pulled.
#: Retryable once, because registries are flaky and a genuinely absent image will
#: fail the retry too and stop there.
EXIT_RUNTIME_ENV = 177

#: A Ray-side failure this module does not recognize. Deliberately *outside* the
#: retryable band: retrying an unknown framework error is how a run spends an
#: hour discovering the same thing three times. Report it and stop.
EXIT_FRAMEWORK = 174

#: Ray exception class name -> exit code. Keyed on the name rather than the class
#: so this table can be built, read and unit-tested without importing Ray, which
#: is what lets `tests/nextflow-genomics-on-ray/test_nf_ray.py` run in CI with no
#: cluster and no Ray install.
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

#: Codes Nextflow's nf-core `base.config` retries. Kept here so the unit test can
#: assert every code this module can emit is on the intended side of that line,
#: rather than trusting a comment.
NFCORE_RETRYABLE: frozenset[int] = frozenset(
    list(range(130, 146)) + [104] + list(range(175, 178))
)


def exit_code_for(exc: BaseException) -> int:
    """The exit code to record for a Ray-side failure.

    Walks the exception's MRO by class name, so a subclass Ray adds later still
    lands on its parent's mapping instead of falling through to
    :data:`EXIT_FRAMEWORK`.
    """
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
