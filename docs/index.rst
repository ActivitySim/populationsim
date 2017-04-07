.. ActivitySim documentation master file, created by
   sphinx-quickstart on Tue May 26 14:13:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PopulationSim
=============

PopulationSim is an open platform for population synthesis.  It emerged
from Oregon DOT's desire to build a shared, open, platform that could be
easily adapted for statewide, regional, and urban transporation planning
needs.

Additional information about the PopulationSim development effort is on the
`GitHub project wiki <https://github.com/rsg/polulationsim/wiki>`__.

Software Design
---------------

PopulationSim is
implemented in Python, and makes heavy use of the vectorized backend C/C++ libraries in 
`pandas <http://pandas.pydata.org>`__  and `numpy <http://numpy.org>`__.  The core design 
principle of the system is vectorization of for loops, and this principle 
is woven into the system wherever reasonable.  As a result, the Python portions of the software 
can be thought of as more of an orchestrator, data processor, etc. that integrates a series of 
C/C++ vectorized data table and matrix operations.  The model system formulates 
each simulation as a series of vectorized table operations and the Python layer 
is responsible for setting up and providing expressions to operate on these large data tables.

Contents
--------

.. toctree::
   :maxdepth: 2

   gettingstarted


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
