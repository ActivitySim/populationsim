.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PopulationSim
=============

PopulationSim is an open platform for population synthesis.  It emerged
from Oregon DOT's desire to build a shared, open, platform that could be
easily adapted for statewide, regional, and urban transportation planning needs.
      
Model Steps
-----------
The example PopulationSim model run consists of the following steps:

* input_pre_processor
* setup_data_structures
* initial_seed_balancing
* meta_control_factoring
* final_seed_balancing
* integerize_final_seed_weights
* sub_balancing.geography=TRACT
* sub_balancing.geography=TAZ
* expand_households
* write_results
* summarize

Each step is described in detail below.

input_pre_processor
~~~~~~~~~~~~~~~~~~~
The inputs pre-processor reads each input table, runs pandas expressions (_expressions.csv) 
against the table to create additional required table fields, and save the tables to the datastore.  
For example, it processes raw Census tables to create the required fields for population 
synthesis.  The inputs pre-processor exposes all input tables to the expressions calculator 
so tables can be joined (such as households to persons for example). It reads the geographic 
crosswalk file in order to join meta, mid, and low level zone tables if needed.  The format of 
the expressions file follows ActivitySim, as shown in the example below.  The ``seed_households`` 
expressions file below operates on the ``seed_households`` input file and processes the ``NPF`` field 
to create the ``FAMTAG`` field, which is then used by PopulationSim in later steps.

+---------------------------------+-------------------------------+-----------------------+
| Description                     |  Target                       |     Expression        |
+=================================+===============================+=======================+
| HH is a family                  |  FAMTAG                       | pd.notnull( NPF ) * 1 |
+---------------------------------+-------------------------------+-----------------------+

setup_data_structures
~~~~~~~~~~~~~~~~~~~~~
setup geographic correspondence, seeds, control sets, weights, expansion factors, and incidence tables

initial_seed_balancing
~~~~~~~~~~~~~~~~~~~~~~
seed balancing

meta_control_factoring
~~~~~~~~~~~~~~~~~~~~~~
meta level control factoring

final_seed_balancing
~~~~~~~~~~~~~~~~~~~~
final balancing for each seed zone with aggregated low and mid-level controls and distributed meta-level controls

integerize_final_seed_weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for each seed zone integerize the household weights

sub_balancing.geography
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
iteratively loop through seed zones, balance all lower level zones within the seed zone, and integerize the resulting household weights

::
 
  sub_balancing.geography=TRACT
  sub_balancing.geography=TAZ

expand_households
~~~~~~~~~~~~~~~~~
expand household and person records with final weights to one household and one person record per weight with unique IDs

write_results
~~~~~~~~~~~~~
write the expanded household and person files to CSV files

summarize
~~~~~~~~~
write summary files - balancer and integerizer results by geography, etc.


Contents
--------

.. toctree::
   :maxdepth: 2
   
   software


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
