import subprocess
from pathlib import Path

Path("candidates").mkdir(exist_ok=True)

for seed in range(10):
    subprocess.run([
        "python", "run_demo.py",
        "--seed", str(seed),
        "--output", f"candidates/run_{seed:02d}.gif"
    ])
