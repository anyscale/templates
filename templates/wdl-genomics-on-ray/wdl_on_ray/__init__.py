"""Run WDL pipelines on Ray.

This package plugs a Ray-backed task dispatcher into `miniwdl
<https://github.com/chanzuckerberg/miniwdl>`_. miniwdl stays responsible for
all the language work (parsing, type checking, expression evaluation,
scatter/conditional expansion, call caching, input localization, output
collection), and this package replaces only the piece that decides *where* a
task's command runs: instead of a subprocess on the machine running the
workflow, each WDL task becomes a Ray task, so it can land on any node of a
(possibly autoscaling) Ray or Anyscale cluster.

The integration point is miniwdl's ``miniwdl.plugin.container_backend`` entry
point, registered here under the name ``ray``. See :mod:`wdl_on_ray.backend`.
"""

from wdl_on_ray._version import __version__

__all__ = ["__version__"]
