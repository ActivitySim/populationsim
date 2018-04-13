.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Software Implementation 
=======================

This page describes the PopulationSim software implementation and how to contribute to PopulationSim.  

The implementation starts with 
the ActivitySim framework, which serves as the foundation for the software.  The framework, as briefly described
below, includes features for data pipeline management, expression handling, testing, etc.  Built upon the 
framework are additional core components for population synthesis such as balancers and integerizers.   
Built upon the population synthesis core components are the model steps that make up a PopulationSim run, 
such as the inputs pre-processor, setting up the data strucutres, doing the initial seed balancing, etc.

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
    
* Expressions

  * Model expressions are in CSV files and contain Python expressions, mainly pandas/numpy expression that operate on the input data tables. This helps to avoid modifying Python code when making changes to the model calculations. 
    
* `Code Documentation <https://activitysim.github.io/activitysim/development.html>`__

  * Python code according to `pycodestyle <https://pypi.python.org/pypi/pycodestyle>`__ style guide
  * Written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`__ markup, built with `Sphinx <http://www.sphinx-doc.org/en/stable>`__ and docstrings written in `numpydoc <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__
    
* `Testing <https://activitysim.github.io/activitysim/development.html>`__

  * A protected master branch that can only be written to after tests have passed
  * `pytest <https://docs.pytest.org/en/latest/>`__ for tests
  * `TravisCI <https://travis-ci.org>`__ for building and testing with each commit

PopulationSim also requires an optimization library for balancing and integerizing.  The software makes 
use of the open source and easy to install `ortools <https://github.com/google/or-tools>`__ package.  The
ortools integerization results varies from platform to platform since edge case results depend on the 
exact ortools/cbc version.

.. _core_components :

Core Components
---------------

assign
^^^^^^

.. automodule:: populationsim.assign
   :members:

balancer
^^^^^^^^

.. automodule:: populationsim.balancer
   :members:

integerizer
^^^^^^^^^^^

.. automodule:: populationsim.integerizer
   :members:

lp
^^
   
.. automodule:: populationsim.lp
   :members:
   
lp_cvx
^^^^^^
   
.. automodule:: populationsim.lp_cvx
   :members:

lp_ortools
^^^^^^^^^^

.. automodule:: populationsim.lp_ortools
   :members:
   
multi_integerizer
^^^^^^^^^^^^^^^^^

.. automodule:: populationsim.multi_integerizer
   :members:

simul_balancer
^^^^^^^^^^^^^^

.. automodule:: populationsim.simul_balancer
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

.. automodule:: populationsim.steps.write_tables
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


Release Notes
-------------

  * v0.3 - first release
  * v0.3.1 - allow zones with zero households
  * v0.3.2 - fix bug in mult-integerizer with total_hh_parent_control_index
  * v0.3.3 - add disgnostic printouts on assert fail in mult_integerizer