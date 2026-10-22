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

PopulationSim requires uv 0.13.4 or newer. To update an existing installation,
see :ref:`certificate_security_update`.

On macOS or Linux:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

On Windows, run this command in PowerShell:

.. code-block:: powershell

   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Follow the installer's instructions to make ``uv`` available in your terminal.
See the `uv installation guide <https://docs.astral.sh/uv/getting-started/installation/>`__
for other installation methods.

.. _certificate_security_update:

Certificate security update
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `October 2026 eMudhra/emSign certificate incident
<https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576>`__
requires uv 0.13.4 or newer, certifi 2026.10.19 or newer, and pip 26.2.2 or newer
where pip is installed, because older versions trust a compromised certificate
authority. Updating certifi alone does not update the certificate bundles
embedded in pip or uv.

With an older uv, commands in the source checkout report an error such as::

   error: Required uv version `>=0.13.4` does not match the running version ...

If you installed uv with the standalone installer, update it with:

.. code-block:: bash

   uv self update --system-certs

``--system-certs`` makes uv check the download with your operating system's
certificates instead of its own outdated list. Note this assumes your system's
CA certificates are up to date. Use it instead of the plain ``uv self update``
that the error message suggests. uv versions older than 0.11 report an error
for this flag. In that case, or if you installed uv another way, rerun the
installer above. Check ``uv --version`` afterwards.

After updating uv and pulling the current checkout, update its environment:

.. code-block:: bash

   uv sync --locked --no-dev

Contributors can use ``uv sync --locked --dev --group docs`` to update their
development and documentation environment. The lockfile includes the fixed
certifi release. For an existing ``uv tool`` installation instead, run:

.. code-block:: bash

   uv tool upgrade populationsim

For pip-managed installations, first upgrade pip using the wheel URL and SHA256
checksum from the Python announcement, then upgrade certifi. Run these commands
with each environment's Python, including newly created virtual environments
that seed an older pip:

.. code-block:: bash

   python -m pip install "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
   python -m pip install --upgrade "certifi>=2026.10.19"

If uv was installed with pip, also upgrade it with
``python -m pip install --upgrade "uv>=0.13.4"`` after upgrading pip. Rebuild
container images to pick up the package and CA certificate updates.

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
