"""Ray Serve entrypoint. `import_path: serve_app:app` in service.yaml points here.

The kMoL ensemble config path is provided via the KMOL_CONFIG_PATH env var so the
same code serves any trained ensemble without edits.
"""

import os

from src.kmol_ensemble import build_app

CONFIG_PATH = os.environ.get("KMOL_CONFIG_PATH", "configs/ensemble_serve.example.json")

app = build_app(CONFIG_PATH)
