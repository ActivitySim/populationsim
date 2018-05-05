.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _getting_started:
   
Getting Started
===============

This page describes how to install and run PopulationSim with the provided example.

Installation
------------

1. Install `Anaconda Python 2.7 <https://www.continuum.io/downloads>`__.  Anaconda Python is required for PopulationSim.

2. If you access the internet from behind a firewall, then you will need to configure your proxy server. To do so, create a .condarc file in your Anaconda installation folder (i.e. ``C:\ProgramData\Anaconda2``), such as:

::

  proxy_servers:
    http: http://myproxy.org:8080
    https: https://myproxy.org:8080
  ssl_verify: false
 
3. Create and activate an Anaconda environment (basically a Python install just for this project)
  
  * Run ``conda create -n popsim python=2.7``
  * Run ``activate popsim`` (you can re-use the environment on a later date by re-activating it or you can skip this step if you don't want to setup a new Python environment just for PopulationSim)
   
4. Get and install other required libraries, which can be found online.  Run the following commands on the activated conda Python environment:

  * `conda install pytables <http://www.pytables.org/>`__
  * `pip install toolz <http://toolz.readthedocs.org/en/latest>`__
  * `pip install zbox <https://github.com/jiffyclub/zbox>`__
  * `pip install orca <https://synthicity.github.io/orca>`__
  * `pip install openmatrix <https://pypi.python.org/pypi/OpenMatrix>`__
  * `pip install activitysim <https://pypi.python.org/pypi/activitysim>`__
  * `pip install ortools <https://github.com/google/or-tools>`__

5. If you access the internet from behind a firewall, then you will need to configure your proxy server when downloading packages. For example:
     
::

  pip install --trusted-host pypi.python.org --proxy=myproxy.org:8080  activitysim

6. Get and install the PopulationSim package on the activated conda Python environment:

::

  pip install https://github.com/RSGInc/populationsim/zipball/master


Run Examples
------------

  * Before running examples, ensure that Anaconda Python, dependent libraries and PopulationSim package have been installed.
 
  * Download and unzip the `example setups <https://github.com/RSGInc/populationSim_resources/raw/master/example_setup/PopulationSimExampleSetUps.7z>`_ to a folder on your computer. It does not have to be the same directory as your Anaconda or PopulationSim install.

There are two examples for running PopulationSim, created using data from the Corvallis-Albany-Lebanon Modeling (CALM) region in Oregon. The `example_calm`_ set-up runs PopulationSim in base mode, where a synthetic population is created for the entire modeling region. This takes approximately 12 minutes on a laptop with an Intel i7-4800MQ CPU @ 2.70GHz and 16 GB of RAM. The `example_calm_repop`_ set-up runs PopulationSim in the *repop* mode, which updates the synthetic population for a small part of the region. More information on the configuration of PopulationSim can be found in the **Application & Configuration** section.

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