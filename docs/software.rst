.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Software Implementation
=======================

This page describes the PopulationSim software implementation and how to contribute to PopulationSim.

The implementation starts with
the ActivitySim framework, which serves as the foundation for the software.  The framework, as briefly described
below, includes features for data pipeline management, expression handling, multiprocessing, testing, etc.  Built upon
the framework are additional core components for population synthesis such as balancers and integerizers.
Built upon the population synthesis core components are the model steps that make up a PopulationSim run,
such as the inputs pre-processor, setting up the data strucutres, doing the initial seed balancing, etc.

.. _development_and_builds:

Development and builds
----------------------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`__, then run
these commands from the repository root. ``.python-version`` selects the default
development Python; the package supports Python 3.10–3.14.

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

ActivitySim Framework
---------------------

PopulationSim is implemented in the `ActivitySim <https://github.com/activitysim/activitysim>`__
framework.  As summarized `here <https://activitysim.github.io/activitysim/#software-design>`__,
being implemented in the ActivitySim framework means:

* Overall Design

  * Implemented in Python, and makes heavy use of the vectorized backend C/C++ libraries in `pandas <http://pandas.pydata.org>`__ and `numpy <http://www.numpy.org>`__.
  * Vectorization instead of for loops when possible
  * Runs sub-models that solve Python expression files that operate on data tables

* Data Handling

  * Inputs are in CSV format, with the exception of settings
  * CSVs are read-in as pandas tables and stored in an intermediate HDF5 binary file that is used for data I/O throughout the model run
  * Key outputs are written to CSV files

* Key Data Structures

  * `pandas.DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`__ - A data table with rows and columns, similar to an R data frame, Excel worksheet, or database table
  * `pandas.Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`__ - a vector of data, a column in a DataFrame table or a 1D array
  * `numpy.array <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`__ - an N-dimensional array of items of the same type, such as a matrix

* Model Orchestrator

  * `ORCA <https://github.com/UDST/orca>`__ is used for running the overall model system and for defining dynamic data tables, columns, and injectables (functions). ActivitySim wraps ORCA functionality to make a Data Pipeline tool, which allows for re-starting at any model step.
  * Support for `multiprocessing <http://docs.python.org/3/library/multiprocessing.html>`_ to reduce runtime

* Expressions

  * Model expressions are in CSV files and contain Python expressions, mainly pandas/numpy expression that operate on the input data tables. This helps to avoid modifying Python code when making changes to the model calculations.

* `Code Documentation <https://activitysim.github.io/activitysim/development.html>`__

  * Python code checked with `Ruff <https://docs.astral.sh/ruff/>`__
  * Written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`__ markup, built with `Sphinx <http://www.sphinx-doc.org/en/stable>`__ and docstrings written in `numpydoc <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__

* `Testing <https://activitysim.github.io/activitysim/development.html>`__

  * A protected master branch that can only be written to after tests have passed
  * `pytest <https://docs.pytest.org/en/latest/>`__ for tests
  * `GitHub Actions <https://github.com/ActivitySim/populationsim/actions>`__ for building and testing with each commit

PopulationSim also requires an optimization library for balancing and integerizing.  The software makes
use of the open source and easy to install `ortools <https://github.com/google/or-tools>`__ package.  The
ortools integerization results varies from platform to platform since edge case results depend on the
exact ortools/cbc version.

.. _core_components :

Core Components
---------------

assign
^^^^^^

.. automodule:: populationsim.core.assign
   :members:

balancer
^^^^^^^^

.. automodule:: populationsim.balancing.single_balancer
   :members:

integerizer
^^^^^^^^^^^

.. automodule:: populationsim.integerizing.single_integerizer
   :members:

lp
^^

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

multi_integerizer
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

PopulationSim development follows the same `development guidelines <https://activitysim.github.io/activitysim/development.html>`__ as ActivitySim.
