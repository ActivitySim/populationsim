.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Software Implementation 
=======================

PopulationSim is implemented in the `ActivitySim <https://github.com/UDST/activitysim>`__ 
framework.  As summarized `here <https://udst.github.io/activitysim/#software-design>`__, 
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
    
* `Code Documentation <https://udst.github.io/activitysim/development.html>`__

  * Python code according to `pep8 <http://legacy.python.org/dev/peps/pep-0008>`__ style guide
  * Written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`__ markup, built with `Sphinx <http://www.sphinx-doc.org/en/stable>`__ and docstrings written in `numpydoc <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__
    
* `Testing <https://udst.github.io/activitysim/development.html>`__

  * A protected master branch that can only be written to after tests have passed
  * `pytest <https://udst.github.io/activitysim/development.html>`__ for tests
  * `TravisCI <https://travis-ci.org>`__ for building and testing with each commit

PopulationSim also requires an optimization library for balancing and integerizing.  Currently the software makes 
use of the open source and easy to install `ortools <https://github.com/google/or-tools>`__ package.

Dependencies
------------
1. If you access the internet from behind a firewall, then you will need to configure your proxy server. To do so, create a .condarc file in your Anaconda installation folder (i.e. ``C:\ProgramData\Anaconda2``), such as:

::

  proxy_servers:
    http: http://proxynew.odot.state.or.us:8080
    https: https://proxynew.odot.state.or.us:8080
  ssl_verify: false

2. Install `Anaconda Python 2.7 <https://www.continuum.io/downloads>`__
3. Create and activate an Anaconda environment (basically a Python install just for this project)
  
  * Run ``conda create -n popsimtest python=2.7``
  * Run ``activate popsimtest`` #you can re-use the environment on a later date by re-activating it
  
4.  If you access the internet from behind a firewall, then you will need to configure your proxy server when downloading packages. For example:

::

  pip install --trusted-host pypi.python.org --proxy=proxynew.odot.state.or.us:8080  activitysim
 
5. Get and install other required libraries, which can be found on the `Python Package Index <https://pypi.python.org/pypi>`__.  Run the following command on the activated conda Python environment: ``pip install <packagename>`` for each package.

  *  `toolz <http://toolz.readthedocs.org/en/latest>`__
  * `zbox <https://github.com/jiffyclub/zbox>`__
  * `orca <https://synthicity.github.io/orca>`__ - the application orchestration tool (i.e. the module runner)
  * `openmatrix <https://pypi.python.org/pypi/OpenMatrix/0.2.3>`__ - OMX support
  * `activitysim <https://pypi.python.org/pypi/activitysim/0.3.dev1>`__ - ActivitySim framework
  * `ortools <https://github.com/google/or-tools>`__ - Google operations research tools solver

Set Up and Run
--------------

* Setup / Installation

  * Setup Python and the dependent libraries as noted above
  * Clone the repository to your computer
  * Cd to the repository folder
  * Run ``activate popsimtest``
  * Run ``python setup.py develop`` to install the package 
  
* Run the existing CALM data example

  * Cd to the ``example_calm`` folder
  * Run ``python run_populationsim.py``
  * Takes approximately 30 minutes to run


Folder & File Setup
-------------------
The example folder is setup as follows:

* run_populationsim.py - runs PopulationSim, which runs the steps described above
* configs folder - configuration settings

  * settings.yaml - settings such as input table names, table key fields, expression files, etc.  The ``geographies`` setting specifies the system geographies in order from largest to smallest - meta, seed, first lower, second lower, etc.  For example, ``[REGION, PUMA, TRACT, TAZ]`` means REGION is the meta geography, PUMA is the seed geography, TRACT is the first lower geography, and TAZ is the second lower geography.  Any number of lower geographies is allowed.  Here is an [example settings file](#settings-file).
  * controls.csv - control variables input file which specifies the controls used for expanding the population.  Like ActivitySim, Python expressions are used for specifying control constraints.  Here is an [example control file](#controls-file).

* data folder - scenario input files

  * seed households records table
  * seed persons records table
  * meta zones data table (i.e. controls and other data)
  * lower level zone data tables (i.e. controls and other data)
  * geographic crosswalk
  
* outputs folder - key outputs

  * households.csv - expanded households
  * persons.csv - expanded persons
  * pipeline.hdf5 - HDF5 data pipeline / data store
  * summary / trace files


Example Settings File
~~~~~~~~~~~~~~~~~~~~~

:: 

  #setup 
  control_file_name: controls.csv
  household_weight_col: WGTP
  household_id_col: hh_id
  total_hh_control: num_hh
  max_expansion_factor: 30
  data_dir: data_calm
  trace_geography:
    TAZ: 100
    TRACT: 10200
  
  #geography setup
  geographies: [REGION, PUMA, TRACT, TAZ]
  seed_geography: PUMA
  crosswalk_data_table: geo_cross_walk
  geography_settings:
    REGION:
      control_data_table: meta_control_data
      id_column: REGION
    PUMA:
      controls_table: seed_controls
      id_column: PUMA
    TRACT:
      control_data_table: mid_control_data
      id_column: TRACT
    TAZ:
      control_data_table: low_control_data
      id_column: TAZ
  
  # input data tables
  input_pre_processor:
    - households
    - persons
    - geo_cross_walk
    - low_control_data
    - mid_control_data
    - meta_control_data
  
  households:
    expression_filename: seed_households_expressions.csv
    filename : seed_households.csv
    index_col: hh_id
    column_map:
      hhnum: hh_id
  
  persons:
    expression_filename: seed_persons_expressions.csv
    filename : seed_persons.csv
    column_map:
      hhnum: hh_id
      SPORDER: per_num
  
  geo_cross_walk:
    filename : geo_cross_walk.csv
    column_map:
      TRACTCE: TRACT
  
  low_control_data:
    filename : control_totals_maz.csv
  
  mid_control_data:
    filename : control_totals_taz.csv
  
  meta_control_data:
    filename : control_totals_meta.csv
  
  #output files
  expanded_households:
    filename : households.csv
    output_fields : serialno, np, nwrkrs_esr, hincp
  expanded_persons:
    filename : persons.csv
    output_fields : sporder, agep, relp, employed

Example Controls File
~~~~~~~~~~~~~~~~~~~~~
The control variables input file specifies the controls used for expanding the population.  Like ActivitySim, Python expressions are used for specifying control constraints.  An example file is below.  

+---------------------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| target                          | geography | seed_table | importance | control_field |  expression                                                         |
+=================================+===========+============+============+===============+=====================================================================+
| num_hh                          | TAZ       | households | 1000000000 | HHBASE        | (households.WGTP > 0) & (households.WGTP < np.inf)                  |
+---------------------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_size_4_plus                  | TAZ       | households | 5000       | HHSIZE4       | households.NP >= 4                                                  |
+---------------------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_age_15_24                    | TAZ       | households | 500        | HHAGE1        | (households.AGEHOH > 15) & (households.AGEHOH <= 24)                |
+---------------------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_inc_15                       | TAZ       | households | 500        | HHINC1        | (households.HHINCADJ > -999999999) & (households.HHINCADJ <= 21297) |
+---------------------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| students_fam_housing            | TAZ       | persons    | 500        | OSUFAM        | persons.OSUTAG == 1                                                 |
+---------------------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_wrks_3_plus                  | TRACT     | households | 1000       | HHWORK3       | households.NWESR >= 3                                               |
+---------------------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_by_type_sf                   | TRACT     | households | 1000       | SF            | households.HTYPE == 1                                               |
+---------------------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| persons_occ_8                   | REGION    | persons    | 1000       | OCCP8         | persons.OCCP == 8                                                   |
+---------------------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+

Where:

* target is the name of the control in PopulationSim
* geography is the geographic level of the control, as specified in ``geographies``
* seed_table is the seed table the control applies to and it can be ``households`` or ``persons``.  If persons, then persons are aggregated to households using the count operator
* importance is the importance weight for the control
* control_field is the field in the control data input files that this control applies to
* expression is a valid Python/Pandas expression that identifies seed households or persons that this control applies to
  
Outputs
-------
The outputs from a PopulationSim run are:

* Synthetic population
  
  * Expanded household and person files in CSV format, as specified in the settings file
  
* System & data management

  * pipeline.h5 - HDF5 data pipeline, with a copy of each table after each model step in pandas format
  * checkpoints.csv - List of data tables in the data pipeline
  * asim.log - log file
  
* Copies of input files

  * control_spec.csv, crosswalk.csv, geo_cross_walk.csv, low_control_data.csv, mid_control_data.csv, meta_control_data.csv, TAZ_controls.csv, TRACT_controls.csv, PUMA_controls.csv, REGION_controls.csv
  
* Household data by model step

  * hh_weights_summary.csv - household weights by model step
  * incidence_table.csv - household control data incidence table
  
* Balanced and intergerized weights for individual households by zone

  * TAZ_weights.csv, TAZ_weights_sparse.csv, TRACT_weights.csv, TRACT_weights_sparse.csv
  
* Aggregate final household totals versus control totals

  * REGION_summary.csv, TAZ_PUMA_summary.csv, TAZ_summary.csv, TRACT_PUMA_summary.csv, TRACT_summary.csv

* Expanded household and person tables

  * households.csv, persons.csv

