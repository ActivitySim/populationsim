.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _getting_started:

Getting Started
===============

This page describes how to install and run PopulationSim with the provided example.

Installation
------------

1. It is recommended that you install and use a *conda* package manager
for your system. One easy way to do so is by using `Anaconda 64bit Python 3 <https://www.anaconda.com/distribution/>`__,
although you should consult the `terms of service <https://www.anaconda.com/terms-of-service>`__
for this product and ensure you qualify (as of summer 2021, businesses and
governments with over 200 employees do not qualify for free usage).  If you prefer
a completely free open source *conda* tool, you can download and install the
appropriate version of `Miniforge <https://github.com/conda-forge/miniforge#miniforge3>`__.

2. If you access the internet from behind a firewall, then you will need to configure your proxy server. To do so, create a .condarc file in your Anaconda installation folder (i.e. ``C:\ProgramData\Anaconda3``), such as:

::

  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false

3. Create and activate an Anaconda environment (basically a Python install just for this project)

::

  conda create -n popsim python=3.8 

  # Windows
  activate popsim

  # Mac
  conda activate popsim

4. Get and install the PopulationSim package on the activated conda Python environment:

::

  # best to use the conda version of pytables for consistency with activitysim
  conda install pytables

  pip install populationsim


.. _activitysim :

ActivitySim
~~~~~~~~~~~

.. note::

  PopulationSim is a 64bit Python 3 library that uses a number of packages from the
  scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__
  and `numpy <http://numpy.org>`__. It also relies heavily on the
  `ActivitySim <https://activitysim.github.io>`__ package.

  The recommended way to get your own scientific Python installation is to
  install 64 bit Anaconda, which contains many of the libraries upon which
  ActivitySim depends + some handy Python installation management tools.

  For more information on Anaconda and ActivitySim, see ActivitySim's `getting started
  <https://activitysim.github.io/activitysim/gettingstarted.html>`__ guide.


Run Examples
------------

There are four examples for running PopulationSim, three created using data from the 
Corvallis-Albany-Lebanon Modeling (CALM) region in Oregon and the other using data from 
the Metro Vancouver region in British Columbia. 

1. The `example_calm`_ set-up runs PopulationSim,  where a synthetic population is created single-processed for the entire modeling region. 

2. The `example_calm_mp`_ set-up runs PopulationSim `multi-processed <http://docs.python.org/3/library/multiprocessing.html>`_, where a synthetic population is created for the entire modeling region by simultaneously balancing results using multiple processors on your computer, thereby reducing runtime.

3. The `example_calm_repop`_ set-up runs PopulationSim in the *repop* mode, which updates the synthetic population for a small part of the region. 

4. The `example_survey_weighting`_ set-up runs PopulationSim for the case of developing final weights for a household travel survey. More information on the configuration of PopulationSim can be found in the **Application & Configuration** section.

Example_calm
~~~~~~~~~~~~

Follow the steps below to run **example_calm** set up:

  * Open a command prompt in the example_calm folder
  * Run the following commands:

  ::

   activate popsim
   python run_populationsim.py

  * Review the outputs in the *output* folder

Example_calm_mp
~~~~~~~~~~~~~~~

Follow the steps below to run **example_calm_mp** multiprocessed set up:

  * Open a command prompt in the example_calm folder
  * In ``configs_mp\setting.yaml``, set ``num_processes: 2`` to a reasonable number of processors for your machine
  * Run the following commands:

  ::

   activate popsim
   python run_populationsim.py -c configs_mp -c configs

  * Review the outputs in the *output* folder

Example_calm_repop
~~~~~~~~~~~~~~~~~~

The repop configuration requires outputs from a base run. Therefore, the base configuration must be run before running the repop configuration. Follow the steps below to run **example_calm_repop** set up:

  * Copy the **pipeline.h5** file from the example_calm\\output directory to example_calm_repop\\output directory (all PopulationSim files are stored in pipeline.h5 file)
  * Open a command prompt in the example_calm_repop folder
  * Run the following commands:

  ::

   activate popsim
   python run_populationsim.py

  * Review the outputs in the *output* folder

Example_survey_weighting
~~~~~~~~~~~~~~~~~~~~~~~~

Follow the steps below to run **example_survey_weighting** set up:

  * Open a command prompt in the example_survey_weighting folder
  * Run the following commands:

  ::

   activate popsim
   python run_populationsim.py

  * Review the outputs in the *output* folder
