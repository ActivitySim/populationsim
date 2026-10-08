.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Software Implementation
=======================

This page describes the PopulationSim software implementation and how to contribute to PopulationSim.

PopulationSim 0.10 includes its own modeling core in ``populationsim.core``,
originally derived from ActivitySim. The ActivitySim package is no longer a
runtime dependency. The core handles configuration, dependency injection,
pipeline checkpoints, expression evaluation, tracing, and multiprocessing.
Population synthesis algorithms live in ``populationsim.balancing`` and
``populationsim.integerizing``, and the model steps live in ``populationsim.steps``.

.. _development_and_builds:

Development and builds
----------------------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`__, then run
these commands from the repository root. ``.python-version`` selects the default
development Python; the source checkout supports Python 3.10–3.14.

.. code-block:: bash

   uv sync --locked --dev
   uv run --locked ruff check .
   uv run --locked pytest --cov=populationsim --cov-report=term-missing
   uv run --locked pre-commit install
   uv build

The Black and Ruff pre-commit hooks use the versions in ``uv.lock``. When changing
dependencies, run ``uv lock`` and commit both ``pyproject.toml`` and ``uv.lock``.

To build the documentation locally:

.. code-block:: bash

   uv sync --locked --group docs --no-dev
   uv run --locked --group docs --no-dev sphinx-build -b html -W --keep-going docs docs/_build/html

Open ``docs/_build/html/index.html`` to view the result.

GitHub Actions runs lint, tests on Linux/macOS/Windows with Python 3.10–3.14,
distribution smoke tests, and documentation builds for pushes and pull requests
to ``master``. Coverage reports, tested distributions, and built HTML are available
as workflow artifacts. A manual run of the Documentation workflow builds HTML for
review. Documentation is published only when a GitHub release is published, and
is built from that release's tagged commit.

The Release workflow uses the same distribution build and smoke tests as CI,
then publishes those artifacts through PyPI trusted publishing when a GitHub
release is published. To release, update the version in ``pyproject.toml``, run
``uv lock``, and merge the changes to ``master``. Manually create and publish a
GitHub release for that commit with a matching ``v`` tag (for example,
``v0.10.0``). Publishing the release automatically triggers both PyPI and
documentation publishing. Both workflows reject tags that do not match the
package version. Pushing a tag or saving a draft release does not publish either
the package or the documentation.

Modeling core and dependencies
------------------------------

The core is maintained within PopulationSim. ``populationsim.core.inject`` wraps
`ORCA <https://github.com/UDST/orca>`__ to register model steps, tables, columns,
and injectable values. ``populationsim.core.pipeline`` executes those steps and
stores table checkpoints so that a run can resume after a completed step.
ORCA remains a required runtime dependency.

Inputs and control specifications are read from CSV files, and configuration is
read from YAML. NumPy arrays and pandas DataFrames and Series hold the model
inputs, weights, and results. Pipeline checkpoints are stored in HDF5 through
pandas and PyTables. Output tables and synthetic populations are generally
written as CSV files; output tables can also be written to HDF5.

``populationsim.balancing`` implements list balancing, including Numba-accelerated
calculations selected with ``USE_NUMBA``. ``populationsim.integerizing`` converts
balanced household weights to integer weights. It uses OR-Tools by default, or
CVXPY when ``USE_CVXPY: True`` is configured. Both solver packages are declared
runtime dependencies and are installed with PopulationSim.

Integerization quantizes weights before rounding by default to reduce sensitivity
to small floating-point differences. Solver versions, configuration, and platform
can still affect solutions; results are not guaranteed to be identical across all
platforms or dependency versions.

The full dependency list and supported Python versions are declared in
`pyproject.toml <https://github.com/ActivitySim/populationsim/blob/master/pyproject.toml>`__.
The repository's ``uv.lock`` records the versions used by its examples and CI.
See :ref:`getting_started` for installation and example commands.

.. _core_components :

Core Components
---------------

assign
^^^^^^

.. automodule:: populationsim.core.assign
   :members:

single_balancer
^^^^^^^^^^^^^^^

.. automodule:: populationsim.balancing.single_balancer
   :members:

single_integerizer
^^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.integerizing.single_integerizer
   :members:

integerization wrappers
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.integerizing.wrappers
   :members:

lp_cvx
^^^^^^

.. automodule:: populationsim.integerizing.lp_cvx
   :members:

lp_ortools
^^^^^^^^^^

.. automodule:: populationsim.integerizing.lp_ortools
   :members:

simul_integerizer
^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.integerizing.simul_integerizer
   :members:

simul_balancer
^^^^^^^^^^^^^^

.. automodule:: populationsim.balancing.simul_balancer
   :members:

.. _model_steps :

Model Steps
-----------

.. _input_pre_processor :

input_pre_processor
^^^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.steps.input_pre_processor
   :members:

.. _setup_data_structures :

setup_data_structures
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.steps.setup_data_structures
   :members:

.. _initial_seed_balancing :

initial_seed_balancing
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.steps.initial_seed_balancing
   :members:

.. _meta_control_factoring :

meta_control_factoring
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.steps.meta_control_factoring
   :members:

.. _final_seed_balancing :

final_seed_balancing
^^^^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.steps.final_seed_balancing
   :members:

.. _integerize_final_seed_weights :


integerize_final_seed_weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.steps.integerize_final_seed_weights
   :members:

.. _sub_balancing :

sub_balancing
^^^^^^^^^^^^^

.. automodule:: populationsim.steps.sub_balancing
   :members:

.. _expand_households :

expand_households
^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.steps.expand_households
   :members:

.. _write_tables :

write_tables
^^^^^^^^^^^^

.. automodule:: populationsim.steps.output
   :members:

.. _write_synthetic_population :

write_synthetic_population
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.steps.write_synthetic_population
   :members:

.. _summarize :

summarize
^^^^^^^^^

.. automodule:: populationsim.steps.summarize
   :members:

.. _repop_balancing :

repop_balancing
^^^^^^^^^^^^^^^

.. automodule:: populationsim.steps.repop_balancing
   :members:

Contribution Guidelines
-----------------------

Create a feature or fix branch from ``master`` and open a pull request targeting
``master``. Include documentation updates when behavior or configuration changes.
Run Ruff, the relevant pytest tests, and the documentation build described in
:ref:`development_and_builds` before submitting the change. CI checks lint,
unit and integration tests, distribution builds, and documentation builds.

When changing dependencies, update ``pyproject.toml`` and regenerate ``uv.lock``
with ``uv lock``. Releases are published manually on GitHub after the changes are
merged; the release workflows then publish the package and documentation.
