"""Run Nextflow pipelines on Ray.

Nextflow's engine is a JVM process, so it cannot call ``ray.remote`` itself. The
integration is therefore split in two, along the seam Nextflow already provides
for HPC schedulers:

* ``nf-ray-plugin/`` -- a Nextflow executor plugin (``executor 'ray'``) that
  subclasses ``AbstractGridExecutor``. It writes ``#RAY`` directives into the job
  script the same way the SLURM executor writes ``#SBATCH``, then submits by
  running ``nf-ray submit``, polls with ``nf-ray status`` and cancels with
  ``nf-ray kill``. That is the whole contract: three commands and a header.

* this package -- the Ray side of those three commands.

The seam matters. Nextflow's grid-executor contract is long-stable, whereas Ray's
Java API is documented as "experimental and only supported by the community" and
would have to load a JNI library inside Nextflow's pf4j plugin classloader.
Keeping Ray in Python costs one ``fork``/``exec`` per submit and keeps the plugin
small.

Ownership is the one subtlety. A Ray task is owned by the process that submitted
it and is cancelled when that process exits, so ``nf-ray submit`` cannot submit
directly -- it would hand Ray a task and then immediately die. Instead a single
long-lived daemon (:mod:`nf_ray.daemon`) holds the Ray connection for the whole
pipeline run and owns every task; the CLI is a stdlib-only client that talks to
it over a unix socket, which is also what keeps ``import ray`` off the
per-submit path.
"""

from nf_ray._version import __version__

__all__ = ["__version__"]
