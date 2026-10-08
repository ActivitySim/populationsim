"""Check an installed distribution's imports and both command-line entry points."""

import shutil
import subprocess
import sys
from importlib.metadata import version
from tempfile import TemporaryDirectory

import populationsim

assert callable(populationsim.run)
assert callable(populationsim.add_run_args)
cli = shutil.which("populationsim")
assert cli is not None, "The populationsim console script was not installed"

with TemporaryDirectory() as working_dir:
    for command in (
        [cli, "--help"],
        [sys.executable, "-I", "-m", "populationsim", "--help"],
    ):
        result = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert "usage:" in result.stdout.lower(), result.stdout

print(f"Smoke test succeeded for PopulationSim {version('populationsim')}")
