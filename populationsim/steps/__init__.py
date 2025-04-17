# PopulationSim
# See full license in LICENSE.txt.

from pathlib import Path
import importlib

from populationsim.core import inject

# Dynamically import all modules in the "steps" folder
for filename in Path(__file__).parent.glob("*.py"):
    if filename.suffix != ".py" or filename.name == "__init__.py":
        continue

    module_name = f"populationsim.steps.{filename.stem}"
    importlib.import_module(module_name)


@inject.injectable(cache=True)
def preload_injectables():
    return True
