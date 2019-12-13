.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _getting_started:

Getting Started
===============

This page describes how to install and run PopulationSim with the provided example.

Installation
------------

1. Install `Anaconda 64bit Python 3 <https://www.anaconda.com/distribution/>`__. Anaconda Python is required for PopulationSim.

2. If you access the internet from behind a firewall, then you will need to configure your proxy server. To do so, create a .condarc file in your Anaconda installation folder (i.e. ``C:\ProgramData\Anaconda3``), such as:

::

  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false

3. Create and activate an Anaconda environment (basically a Python install just for this project)

::

  conda create -n popsim python=3.7

  #Windows
  activate popsim

  #Mac
  conda activate popsim

4. Get and install the PopulationSim package on the activated conda Python environment:

::

  pip install populationsim


.. _anaconda_notes :

Python 2 or 3?
~~~~~~~~~~~~~~~

.. note::

  PopulationSim is a 64bit Python 2 or 3 library that uses a number of packages from the
  scientific Python ecosystem, most notably `pandas <http://pandas.pydata.org>`__
  and `numpy <http://numpy.org>`__. It relies heavily on the
  `ActivitySim <https://activitysim.github.io>`__ package. Both ActivitySim and PopulationSim
  currently support Python 2, but Python 2 will be `retired <https://pythonclock.org/>`__ at the
  end of 2019 so Python 3 is recommended.

  The recommended way to get your own scientific Python installation is to
  install 64 bit Anaconda, which contains many of the libraries upon which
  ActivitySim depends + some handy Python installation management tools.

  For more information on Anaconda and ActivitySim, see ActivitySim's `getting started
  <https://activitysim.github.io/activitysim/gettingstarted.html#anaconda>`__ guide.


Run Examples
------------

  * Before running examples, ensure that Anaconda Python, dependent libraries and PopulationSim package have been installed.

  * Download and unzip the `example setups <https://github.com/RSGInc/populationSim_resources/raw/master/example_setup/PopulationSimExampleSetUpsPython3.zip>`_ to a folder on your computer. It does not have to be the same directory as your Anaconda or PopulationSim install.

There are three examples for running PopulationSim, two created using data from the Corvallis-Albany-Lebanon Modeling (CALM) region in Oregon and the other using data from the Metro Vancouver region in British Columbia. The `example_calm`_ set-up runs PopulationSim in base mode, where a synthetic population is created for the entire modeling region. This takes approximately 12 minutes on a laptop with an Intel i7-4800MQ CPU @ 2.70GHz and 16 GB of RAM. The `example_calm_repop`_ set-up runs PopulationSim in the *repop* mode, which updates the synthetic population for a small part of the region. The `example_survey_weighting`_ set-up runs PopulationSim for the case of developing final weights for a household travel survey. More information on the configuration of PopulationSim can be found in the **Application & Configuration** section.

Example_calm
~~~~~~~~~~~~

Follow the steps below to run **example_calm** set up:

  * Open a command prompt in the example_calm folder
  * Run the following commands:

  ::

   activate popsim
   python run_populationsim.py

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

Example_weighting
~~~~~~~~~~~~~~~~~~

Follow the steps below to run **example_survey_weighting** set up:

  * Open a command prompt in the example_survey_weighting folder
  * Run the following commands:

  ::

   activate popsim
   python run_populationsim.py

  * Review the outputs in the *output* folder
