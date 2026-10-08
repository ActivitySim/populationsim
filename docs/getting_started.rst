.. _getting_started:

Getting Started
===============

The current source checkout supports Python 3.10–3.14. These instructions use
`uv <https://docs.astral.sh/uv/>`__ to manage Python, environments, and dependencies.
PopulationSim includes its own modeling core and runs independently of the
ActivitySim package. Its runtime dependencies, including ORCA, are installed
automatically.

Installation
------------

Install uv
~~~~~~~~~~

On macOS or Linux:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

On Windows, run this command in PowerShell:

.. code-block:: powershell

   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Follow the installer's instructions to make ``uv`` available in your terminal.
See the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`__
for other installation methods.

Install PopulationSim and the examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install `Git <https://git-scm.com/downloads>`__, then clone the repository and
install its runtime dependencies:

.. code-block:: bash

   git clone https://github.com/ActivitySim/populationsim.git
   cd populationsim
   uv sync --locked --no-dev
   uv run --locked --no-dev populationsim --help

uv creates ``.venv`` in the repository, uses the Python version selected by
``.python-version``, and downloads Python if needed. ``--locked`` uses the committed
dependency versions in ``uv.lock``; ``--no-dev`` omits development tools. Run the
remaining example commands from this repository root. ``uv run`` uses the project
environment, so you do not need to activate it.

Cloning the default branch installs the current development version. For the
published package, use the command-line installation below. For contributor
setup, tests, and documentation builds, see
:ref:`development_and_builds`.

Install the released command-line application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run your own project with the package published on PyPI, you can install the
command-line application in a separate environment managed by uv:

.. code-block:: bash

   uv tool install --python 3.12 populationsim
   populationsim --help

If uv reports that its executable directory is missing from ``PATH``, run
``uv tool update-shell`` and reopen your terminal. This installation provides the
``populationsim`` command; it does not download the repository's example data.

With existing config, data, and output directories, run:

.. code-block:: bash

   populationsim -c /path/to/configs -d /path/to/data -o /path/to/output

Run Examples
------------

The examples below use the source checkout and ``uv run`` setup above. The bundled
examples include ``output`` directories. When creating your own project, create
its output directory before running: the application checks that config, data,
and output directories exist. Review the generated CSV files, logs, and pipeline
checkpoints in each example's ``output`` directory.

.. _example_calm:

CALM: single process
~~~~~~~~~~~~~~~~~~~~

``examples/example_calm`` synthesizes a population for the
Corvallis-Albany-Lebanon Modeling (CALM) region in Oregon:

.. code-block:: bash

   uv run --locked --no-dev populationsim -w examples/example_calm

This run creates the ``pipeline.h5`` checkpoint needed by the CALM repopulation
example below.

.. _example_calm_mp:

CALM: multiprocessing
~~~~~~~~~~~~~~~~~~~~~

The multiprocessing configuration is in ``examples/example_calm/configs_mp``;
it uses the same data and output directory as the single-process example:

.. code-block:: bash

   uv run --locked --no-dev populationsim -w examples/example_calm -m 2

``-m 2`` enables multiprocessing and adds ``configs_mp`` before ``configs`` in
the configuration search order. Review ``configs_mp/settings.yaml`` when adjusting
the number of processes: it contains both the overall ``num_processes`` setting
and a per-step override. This configuration keeps only the final checkpoint in
``final_pipeline.h5`` and removes the working pipeline files. Run the
single-process example again before preparing the repopulation example, which
needs the earlier ``summarize`` checkpoint.

.. _example_calm_repop:

CALM: repopulation
~~~~~~~~~~~~~~~~~~

``examples/example_calm_repop`` updates part of an existing synthetic population.
First complete the single-process CALM run above, then copy its checkpoint into
the repopulation output directory:

.. code-block:: bash

   uv run --locked --no-dev python -c "import shutil; shutil.copyfile('examples/example_calm/output/pipeline.h5', 'examples/example_calm_repop/output/pipeline.h5')"
   uv run --locked --no-dev populationsim -w examples/example_calm_repop

The configuration resumes from the base run's ``summarize`` checkpoint. Copy a
fresh base checkpoint when you want to repeat the repopulation run from the
original population.

.. _example_survey_weighting:

Survey weighting
~~~~~~~~~~~~~~~~

``examples/example_survey_weighting`` develops final household travel survey
weights rather than expanding seed households into a synthetic population:

.. code-block:: bash

   uv run --locked --no-dev populationsim -w examples/example_survey_weighting

See :doc:`application_configuration` for input files, controls, and model settings.
The repository also includes ``example_oceanside_repop`` and ``example_test``;
the latter provides a small dataset used by the test suite.
