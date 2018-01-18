.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />
   
Application & Configuration
=============================

This section describes how to set up a new PopulationSim implementation. The first step is to understand the requirements of the project in terms of geographic resolution and details desired in the synthetic population. Once the requirements of the project have been established, the next step is to prepare the inputs to PopulationSim. Next, PopulationSim needs to be configured for available inputs and features desired in the final synthetic population. After this, the user needs to run PopulationSim and resolve any data related errors. Finally, the output synthetic population needs to be validated against the controls for precision and variance.

Selecting Geographies
----------------------

Traditionally, travel forecasting models have followed the sequential four-step model framework. This required the modeling region to be divided into zones. The 4-step forecasting process starts with *trip generation* in each zone using the available demographic data. Next, *trip distribution* between zones, and finally, *mode choice* and *route assignment*. The zones used in four-step process are typically known as Traffic Analysis Zones (TAZs). The spatial boundaries of TAZs varies across modeling region and ranges from a city block to a large area in the suburb within a modeling region. Smaller TAZs result in additional zones and thus adds to the computational burden. 

ABMs on the other hand are computationally efficient and operate in a micro-simulation framework at the level of persons and households. Most of the advanced ABMs (e.g., DaySim, CT-RAMP) operate at a finer spatial resolution wherein all location choices (e.g., usual work location, tour destination choice) are modeled at a finer geography. This finer geography typically is the Micro-Analysis Zones (MAZs) which are smaller zones nested within TAZs. This requires the synthetic population to be produced at the level of MAZs.

As discussed earlier, two main inputs to a population synthesizer are a seed sample and controls. The seed sample can come from a household travel survey or from ACS PUMS with latter being the most common source. In case of a household travel survey, the geographic resolution of the seed sample is determined by the geocoding geography. The PUMS data contains a sample of actual responses to ACS. While PUMS data contain records from disaggregate geographies but data is made available at an aggregate geography called Public Use Microdata Area (PUMA). PUMAs are special non-overlapping areas that partition each state into contiguous geographic units containing no fewer than 100,000 people each.

Ideally, it is desired that all the controls and seed sample should be available at the same  level of geographic resolution as the travel model. However, this is typically not the case. Some important demographic, socio-economic and land-use development distributions (e.g., employment or occupation data) that dictates population synthesis are only available at a more aggregate geography (e.g., County, District, Region, etc.). Moreover, some distributions which are available at a finer geographic level in the base year may not be available at the same geographic level for a future forecast year. In some cases, even if a control is available at a finer geography, the modeler might want to specify that control (e.g., population by age) at an aggregate geography.

The flexible number of geographies feature in PopulationSim enables user to make use of data available at different geographic resolutions. In summary, the **choice of geographies for PopulationSim** is decided based on following:

:Travel Model Spatial Resolution:
	For most ABMs, this is **MAZ** but can also be **TAZ** or **Block Group**
	
:Seed Sample Geography:
	Typically, this would be **PUMA** but can be some other geography depending on the source of seed sample
	
:Marginal Controls Geographies: 
	User would need to specify at least one control at the travel model spatial resolution. Geography of other controls would depend upon the availability of data and user preferences. 

The hierarchy of geographies is important when making a decision regarding controls. The hierarchy of geographies in PopulationSim framework is as follows:
	
  * Meta (e.g., County)
  * Seed (e.g., PUMA)
  * Sub-Seed (e.g., TAZ, MAZ)
 
Seed geography is the geographic resolution of the seed data. PopulationSim starts at this geography and moves up to the Meta geography. Currently, PopulationSim can handle only one Meta geography. After incorporating controls at the Meta geography, PopulationSim moves down to Sub-Seed geographies. PopulationSim can handle any number of Sub-Seed geographies. While selecting control geographies user should not select more than one Meta geography. More information on PopulationSim algorithm can be found from the PopulationSim specifications in the **Documents & Resources** section.

Geographic Cross-walk
~~~~~~~~~~~~~~~~~~~~~

After selecting the geographies, the next step is to prepare a geographic cross-walk file. The geographic cross-walk file defines the hierarchical structure of geographies. The geographic cross-walk is used to aggregate controls specified at a lower geography to upper geography and to allocate population from an upper geography to a lower geography.

  
Preparing seed and control data
--------------------------------

Seed sample
~~~~~~~~~~~

As mentioned in previous section, seed sample generally is built from the ACS PUMS. One of the main requirements for the seed sample is that it should be representative of the modeling region. In case of ACS PUMS, this can be ensured by selecting PUMAs representing the modeling region both demographically and geographically. PUMA boundaries may not perfectly line up against the modeling region boundaries and overlaps are possible. Each sub-seed geography should be assigned to a PUMA and each PUMA should be assigned to a Meta geography.

Next important requirement is that seed sample should contain all control variables and also the variables that are needed for the travel model. For completely segmented population groups such as residential population and group-quarter population, separate seed samples are prepared. PopulationSim can be set up and run separately for these two population segments using the same geographic system. The outputs from the two runs can be combined into a unified synthetic population as a post processing step.

Finally, the seed sample should include an initial weight field. PopulationSim algorithm is designed to assign weights as close to the initial weight as possible to minimize the changes in distribution of uncontrolled variables. All the fields in the seed sample should be appropriately recoded to specify controls (see more details in next section). Household-level population variables needs to be computed in advance (for e.g., Number of Workers) and monetary variables should be inflation adjusted (e.g., Household Income)

Controls
~~~~~~~~~

Controls are the marginal distributions that form the constraints for the population synthesis procedure. Controls are also referred to as *targets* and the objective of the population synthesis procedure is to produce a synthetic population whose control fields would match these marginal distributions. Controls can be specified for both household and person variables. The choice of control variables depends on the needs of the project. 

The mandatory requirement for a population synthesizer is to generate the right number of households in each travel model geography. Therefore, it is mandatory to specify a control on total number of households in each geography. This control is specified at the lowest geography and if matched perfectly would ensure that all the upper geographies would also have the correct number of households assigned to them. Ideally, user would want to specify control for all variables that are important determinant of travel behaviour or would be of interest to policy makers. These would include social, demographic, economic and land-use related variables.

There are multiple source to obtain input data to build these controls. The modeling agency may collect important demographic data for the modeling region (e.g., number of households). Some data can also be obtained from a socio-economic or land-use model for the region such as, households by income groups or households by housing type. Most commonly, controls are build from sources such as Census Summary File 1, 2 and 3, ACS PUMS distributions, CTPP, etc. 

The geography at which a control is specified is determined by the travel model geographies or the geography of the source data. Common travel model geographies are TAZ and MAZ. Outputs from agency socio-economic or land-use model might be available at these geographies that can be used to build control at these geographies for available variables. Data from Census sources are usually available at one of the Census geographies - Block, Block Groups, Census Tract, County, etc. Once the data has been downloaded, the next step is to aggregate/disaggregate the data to the desired geography. 

Disaggregation involves distributing data from the upper geography to lower geographies using a distribution based on area, population or number of households. A simple aggregation is possible when the lower geography boundaries fits perfectly within the upper geography boundary. In case of overlaps, data can again be aggregated in proportion to the area. A more common and intuitive method is to establish a correspondence between the lower and upper geography based on the position of the geometric centroid of the lower geography. If the centroid of the lower geography lies within the upper geography then the whole lower geography is assumed to lie within the upper geography. For some shapes, the geometric centroid might be outside the shape boundary. In such cases, an internal point closest to the geometric centroid but within the shape is used. All Census shape files come with the coordinates of the internal point.  The user would need to download the Census shape files for the associated geography and then establish a correspondence with the desired geography using this methodology. It is recommended that input control data should be obtained at the lowest geography possible and then aggregated to the desired geography. 


Configuration
-------------

Below is PopulationSim's directory structure followed by description of inputs. To set up a PopulationSim run, user would need to create this directory structure. A template directory structure can be downloaded from `here <https://resourcesystemsgroupinc-my.sharepoint.com/personal/binny_paul_rsginc_com/_layouts/15/guestaccess.aspx?docid=138e31404fd894713b083135b69707f97&authkey=AWwjSOG61Xu-JB6e9Fx6tMM&expiration=2018-07-14T01%3A21%3A15.000Z&e=CbdvmX>`_

  .. image:: PopulationSimFolderStructure.png

  
PopulationSim is configured to run using the batch file **RunPopulationSim.bat**. User needs to update the path to the Anaconda install (Anaconda2 folder) on their computer. This batch file activates the *populationsim* environment and then calls the *run_populationsim.py* Python script to launch a PopulationSim run. Open the **RunPopulationSim.bat** file in edit mode and change the path to Anaconda install as follows:

::

   :: USER INPUTS
   :: ---------------------------------------------------------------------
   :: Local Anaconda installation directory
   SET ANACONDA_DIR=E:\path\to\this\directory\Anaconda2
   :: ---------------------------------------------------------------------

Two configurations are available to run PopulationSim - **base** and **repop**.

:base configuration:

  base configuration is the default mode and does not require any changes from the user. It runs PopulationSim from beginning to end and produces a new synthetic population. The call to run_populationsim.py script looks as follows:
  
::

   %PYTHON% run_populationsim.py

:repop configuration:

  repop configuration is used for repopulating a subset of zones for an existing synthetic population. User has the option to *replace* or *append* to the existing synthetic population. These options are specified from the settings file, details can be found in the *Configuring Settings File* section. The call to run_populationsim.py script under *repop* mode looks as follows:

::

   %PYTHON% run_populationsim.py -m repop
   
The following sections describes the inputs and outputs, followed by discussion on configuring the settings file and specifying controls. 

Inputs & Outputs
~~~~~~~~~~~~~~~~~~~

Please refer to the following definition list to understand the file names:

:*GEOG_NAME*: Sub-seed geography name such as TAZ, MAZ, etc.
:*SEED_GEOG*: Geographic resolution of the seed sample such as PUMA.
:*META_GEOG*: Geography name of the Meta geography such as Region, District, etc.

 
--------------------------------------------------------------  

Working Directory Contents:

+-----------------------+----------------------------------------------------------------------------+
| File                  | Description                                                                |
+=======================+============================================================================+
| RunPopulationSim.bat  | Batch file to run PopulationSim                                            |
+-----------------------+----------------------------------------------------------------------------+
| run_populationsim.py  | Python script that orchestrates a PopulationSim run                        |
+-----------------------+----------------------------------------------------------------------------+
| /configs              | Sub-directory containing control specifications and configuration settings |
+-----------------------+----------------------------------------------------------------------------+
| /data                 | Sub-directory containing all input files                                   |
+-----------------------+----------------------------------------------------------------------------+
| /output               | Sub-directory containing all outputs, summaries and intermediate files     |
+-----------------------+----------------------------------------------------------------------------+

--------------------------------------------------------------  

*/configs* Sub-directory Contents:

+--------------------+-----------------------------------------------------------------------------------+
| File               | Description                                                                       |
+====================+===================================================================================+
| logging.yaml       | YAML-based file for setting up logging                                            |
+--------------------+-----------------------------------------------------------------------------------+
| settings.yaml      | YAML-based settings file to configure a PopulationSim run                         |
+--------------------+-----------------------------------------------------------------------------------+
| controls.csv       | CSV file to specify controls                                                      |          
+--------------------+-----------------------------------------------------------------------------------+
| repop_controls.csv | CSV file to specify controls when running PopultionSim in the repop configuration |
+--------------------+-----------------------------------------------------------------------------------+

--------------------------------------------------------------  

*/data* Sub-directory Contents:

+-------------------------------------+-------------------------------------------------------------------------------------+
| File                                | Description                                                                         |
+=====================================+=====================================================================================+
| control_totals_GEOG_NAME.csv        | Marginal control totals at each spatial resolution named *GEOG_NAME*                |
+-------------------------------------+-------------------------------------------------------------------------------------+
| repop_control_totals_GEOG_NAME.csv  | Marginal control totals at each spatial resolution named *GEOG_NAME* for repop run  |
+-------------------------------------+-------------------------------------------------------------------------------------+
| geo_crosswalk.csv                   | Geographic cross-walk file                                                          |          
+-------------------------------------+-------------------------------------------------------------------------------------+
| seed_households.csv                 | Seed sample of households                                                           |          
+-------------------------------------+-------------------------------------------------------------------------------------+
| seed_persons.csv                    | Seed sample of persons                                                              |
+-------------------------------------+-------------------------------------------------------------------------------------+

--------------------------------------------------------------  

*/output* Sub-directory Contents (populated at the end of a PopulationSim run):

This sub-directory is populated at the end of the PopulationSim run. The table below list all possible outputs from a PopulationSim run. User has the option to specify the output files that should be exported at the end of a run, details can be found in the *Configuring Settings File* section.

+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| File                            | Group                      | Description                                                                             |
+=================================+============================+=========================================================================================+
| asim.log                        | Logging                    | Log file                                                                                |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| pipeline.h5                     | Data Pipeline              | HDF5 data pipeline which stores all the inputs, outputs and intermediate files          |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| expanded_household_ids.csv      | Final Synthetic Population | List of expanded household IDs with their geographic assignment. User would join |br|   | 
|                                 |                            | this file with the seed sample to generate a fully expanded synthetic population        |          
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| expanded_households.csv         | Final Synthetic Population | Fully expanded synthetic population of households. User can specify the attributes |br| |
|                                 |                            | to be included from the *seed sample* in the *settings.YAML* file                       |           
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| expanded_persons.csv            | Final Synthetic Population | Fully expanded synthetic population of persons. User can specify the attributes to |br| | 
|                                 |                            | be included from the *seed sample* in the *settings.YAML* file                          |          
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| incidence_table.csv             | Intermediate               | Intermediate incidence table                                                            |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| household_groups.csv            | Intermediate               | Unique household group assignments based on controls variables                          |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| GEOG_NAME_control_data.csv      | Intermediate               | Input control data at each geographic level - *GEOG_NAME*                               |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| GEOG_NAME_controls.csv          | Intermediate               | Control totals at each geographic level (*GEOG_NAME*) containing only the controls |br| |
|                                 |                            | specified in the *configs/controls.csv* control specification file                      |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| GEOG_NAME_weights.csv           | Intermediate               | List of household weights with their geographic assignment                              |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| GEOG_NAME_weights_sparse.csv    | Intermediate               | List of household weights with their geographic assignment                              |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| control_spec.csv                | Intermediate               | Control specification used for the run                                                  |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| geo_cross_walk.csv              | Intermediate               | Input geographic cross-walk                                                             |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| crosswalk.csv                   | Intermediate               | Trimmed geographic cross-walk used in PopulationSim run                                 |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| trace_GEOG_NAME_weights.csv     | Tracing                    | Trace file listing household weights for the trace geography specified in settings      |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| summary_hh_weights.csv          | Summary                    | List of household with weights through different stages of PopulationSim                |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| summary_GEOG_NAME.csv           | Summary                    | Marginal Controls vs. Synthetic Population Comparison at *GEOG_NAME* level              |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| summary_GEOG_NAME_aggregate.csv | Summary                    | Household weights aggregate to *SEED_GEOG* at the end of allocation to *GEOG_NAME*      |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+
| summary_GEOG_NAME_SEED_GEOG.csv | Summary                    | Marginal Controls vs. Synthetic Population Comparison at *SEED_GEOG* level using |br|   |
|                                 |                            | weights from allocation at *GEOG_NAME* level                                            |
+---------------------------------+----------------------------+-----------------------------------------------------------------------------------------+


Configuring Settings File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PopulationSim is configured using the *configs/settings.YAML* file. The user has the flexibility to specify the following settings:

**Algorithm/Software Configuration**:

:: 

  INTEGERIZE_WITH_BACKSTOPPED_CONTROLS: True
  SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS: False
  GROUP_BY_INCIDENCE_SIGNATURE: True
  USE_SIMUL_INTEGERIZER: True
  USE_CVXPY: False
  max_expansion_factor: 30

+--------------------------------------+------------+---------------------------------------------------------------------------------+
| Attribute                            | Value      | Description                                                                     |
+======================================+============+=================================================================================+
| INTEGERIZE_WITH_BACKSTOPPED_CONTROLS | True/False | When set to **True**, upper geography controls are imputed for current |br|     |
|                                      |            | geography and used as additional controls for integerization                    |
+--------------------------------------+------------+---------------------------------------------------------------------------------+
| SUB_BALANCE_WITH_FLOAT_SEED_WEIGHTS  | True/False | When **True**, PopulationSim uses floating weights from upper geography         |
+--------------------------------------+------------+---------------------------------------------------------------------------------+
| GROUP_BY_INCIDENCE_SIGNATURE         | True/False | When **True**, PopulationSim groups the household incidence by HH group         |
+--------------------------------------+------------+---------------------------------------------------------------------------------+
| USE_SIMUL_INTEGERIZER                | True/False | PopulationSim Integerizer can run in two modes: |br|                            |
|                                      |            |      1. Sequential - Zones are processed in a ascending order of number of |br| |
|                                      |            |         households in the zone |br|                                             |
|                                      |            |      2. Simultaneous - Zones are processed simultaneously |br|                  |
|                                      |            |                                                                                 |
|                                      |            | *for more details, refer the TRB paper on Docs page*                            |
+--------------------------------------+------------+---------------------------------------------------------------------------------+
| USE_CVXPY                            | True/False | A third-party solver is used for integerization - CVXPY or or-tools |br|        |
|                                      |            | **CVXPY** currently is not available for Windows                                |
+--------------------------------------+------------+---------------------------------------------------------------------------------+
| max_expansion_factor                 | > 0        | Maximum HH expansion factor weight setting. This settings dictates the |br|     |
|                                      |            | ratio of the final weight of the household record to its initial weight. |br|   |
|                                      |            | For example, a maxExpansionFactor setting of 5 would mean a household |br|      |
|                                      |            | having a PUMS weight of x can have a final weight of not more than 5x, |br|     |
|                                      |            | thus effectively restricting the number of times a record can be sampled. |br|  |
|                                      |            | The user might need to adjust this setting to enable sampling of a record |br|  |
|                                      |            | with a rare household configuration. Otherwise, it might result in some |br|    |
|                                      |            | controls not being matched due to unavailability of records to sample from |br| |
+--------------------------------------+------------+---------------------------------------------------------------------------------+

        

**Geographic Settings**:

:: 

  geographies: [META_GEOG, SEED_GEOG, SUB_SEED_GEOG_1, SUB_SEED_GEOG_2]
  seed_geography: SEED_GEOG

+----------------+---------------------+---------------------------------------------------------------------------------+
| Attribute      | Value               | Description                                                                     |
+================+=====================+=================================================================================+
| geographies    | List of geographies | List of geographies at which the controls are specified including the SEED |br| |
|                |                     | geography. The geographies should be in the following order: |br|               |
|                |                     | *META_GEOG* >> *SEED_GEOG* >> *SUB_SEED_GEOG_1* >> *SUB_SEED_GEOG_2* >> ... |br||
|                |                     | Example - [REGION, PUMA, TAZ, MAZ] |br|                                         |
|                |                     | Any number of geographies are allowed |br|                                      |
|                |                     | These geography names should be used as prefixes in control data file names |br||
|                |                     | for the corresponding geographies                                               |
+----------------+---------------------+---------------------------------------------------------------------------------+
| seed_geography | SEED_GEOG           | Seed geography name from the list of geographies                                |
+----------------+---------------------+---------------------------------------------------------------------------------+


**Tracing**:

:: 

  trace_geography:
	GEOG_1: 100
	GEOG_2: 10200

+-----------+---------------------------------------------------------------------------------+
| Attribute | Description                                                                     |
+===========+=================================================================================+
| GEOG_1    | ID of GEOG_1 zone that should be traced. Example, TRACT = 100                   |
+-----------+---------------------------------------------------------------------------------+
| GEOG_2    | ID of GEOG_1 zone that should be traced. Example, TAZ = 10200                   |
+-----------+---------------------------------------------------------------------------------+

**data directory**:

:: 

  data_dir: data

+-----------+---------------------------------------------------------------------------------+
| Attribute | Description                                                                     |
+===========+=================================================================================+
| data_dir  | Name of the data_directory within the working directory                         |
+-----------+---------------------------------------------------------------------------------+


**Input Data Tables**

This setting is used to specify details of various inputs to PopulationSim. Below is the list of the inputs in the PopulationSim data pipeline:

	* Seed-Households
	* Seed-Persons
	* Geographic CrossWalk 
	* Control data at each control geography 
	
For each input table, user is required to specify an import table name, input CSV file name, index column name and column name map (only for renaming column names). User can also specify a list of columns to be dropped. An example is shown below followed by description of attributes.

::

	input_table_list:
	- tablename: households
		filename : seed_households.csv
		index_col: hh_id
		column_map:
		hhnum: hh_id
	- tablename: persons
		filename : seed_persons.csv
		column_map:
		hhnum: hh_id
		SPORDER: per_num
		# drop mixed type fields that appear to have been incorrectly generated
		drop_columns:
		- indp02
		- naicsp02
		- occp02
		- socp00
		- occp10
		- socp10
		- indp07
		- naicsp07
	- tablename: geo_cross_walk
		filename : geo_cross_walk.csv
		column_map:
		TRACTCE: TRACT
	- tablename: TAZ_control_data
		filename : control_totals_taz.csv
	- tablename: TRACT_control_data
		filename : control_totals_tract.csv
	- tablename: REGION_control_data
		filename : scaled_control_totals_meta.csv

+--------------+---------------------------------------------------------------------------------------+
| Attribute    | Description                                                                           |
+==============+=======================================================================================+
| tablename    | Name of the imported CSV file in the PopulationSim data pipeline. The input |br|      |
|              | names in the PopulationSim data pipeline should be named as per the following |br|    |
|              | standard: |br|                                                                        |
|              | 1. Seed-Households - *households* |br|                                                |
|              | 2. Seed-Persons - *persons* |br|                                                      |
|              | 3. Geographic CrossWalk - *geo_cross_walk* |br|                                       |
|              |                                                                                       |
|              |    The field names in the geographic cross-walk should be same as the geography |br|  |
|              |    names specified in the settings file                                               |
|              |                                                                                       |
|              | 4. Control data at each control geography - *GEOG_NAME_control_data*, |br|            |
|              |    where *GEOG_NAME*  is the name of the control geography                            |
|              |                                                                                       |
+--------------+---------------------------------------------------------------------------------------+
| filename     | Name of the input CSV file in the data folder                                         |
+--------------+---------------------------------------------------------------------------------------+
| index_col    | Name of the unique ID field in the seed household data                                |          
+--------------+---------------------------------------------------------------------------------------+
| column_map   | Column map of fields to be renamed. The format for the column map is as follows: |br| |          
|              | ``Name in CSV: New Name``                                                             |
+--------------+---------------------------------------------------------------------------------------+
| drop_columns | List of columns to be dropped from the input data                                     |
+--------------+---------------------------------------------------------------------------------------+


**Reserved Column Names**:

Three columns representing the following needs to be specified:

- Initial weight on households
- Unique household identifier
- Control on total number of households at the lowest geographic level

:: 

  household_weight_col: WGTP
  household_id_col: hh_id
  total_hh_control: num_hh

+------------------------+------------------------------------------------------------------+
| Attribute              | Description                                                      |
+========================+==================================================================+
| household_weight_col   | Initial weight column in the household seed sample               |
+------------------------+------------------------------------------------------------------+
| household_id_col       | Unique household ID column in the household seed sample          |
+------------------------+------------------------------------------------------------------+
| total_hh_control       | Total number of household control at the lowest geographic level |
+------------------------+------------------------------------------------------------------+


**Control Specification File Name**:

::

  control_file_name: controls.csv

+---------------------+--------------------------------------------+
| Attribute           | Description                                |
+=====================+============================================+
| control_file_name   | Name of the CSV control specification file |
+---------------------+--------------------------------------------+


**Output Tables**:

Inputs & Outputs section listed all possible outputs. This setting is used to specify the list of outputs. User can specify either a list of output tables to include or to skip using the *action* attribute as shown below in the example. if neither is specified, then all check pointed tables will be written. The HDF5 data pipeline and all summary files are written out regardless of this setting.

::

  output_tables:
    action: include
    tables:
      - households
      - persons  
	  - expanded_household_ids

+------------+---------------------------------------------------+
| Attribute  | Description                                       |
+============+===================================================+
| action     | *include* or *skip* the list of tables specified  |
+------------+---------------------------------------------------+
| tables     | List of table to be written out or skipped        |
+------------+---------------------------------------------------+
	  
	
**Steps for base mode**:	  

This setting lists the sub-modules or steps to be run by the PopulationSim orchestrator. ActivitySim framework allows user to resume a PopulationSim run from a specific point. This is specified using the attribute ``resume_after``. The step, ``sub_balancing.geography`` is repeated for each sub-seed geography.

::

  run_list:
    steps:
      - input_pre_processor
      - setup_data_structures
      - initial_seed_balancing
      - meta_control_factoring
      - final_seed_balancing
      - integerize_final_seed_weights
      - sub_balancing.geography=TRACT
      - sub_balancing.geography=TAZ
      - expand_households
      - write_results
      - summarize
  
    #resume_after: integerize_final_seed_weights	  
	  
+----------------+---------------------------------------------------+
| Attribute      | Description                                       |
+================+===================================================+
| steps          | List of steps to be run                           |
+----------------+---------------------------------------------------+
| resume_after   | The step from which the current run should resume |
+----------------+---------------------------------------------------+


**Steps for repop mode**:

When running PoulationSim in repop mode, the steps specified in this setting are run. The repop mode runs over an existing synthetic population and uses the data pipeline HDF5 file from the base run as an input. The default value for the ``resume_after`` setting under the repop mode is *summarize* which is the last step of a base run. In other words, repop mode starts from the last step of the base run and modifies the base synthetic population as per the new controls. User can choose either *append* or *replace* in the ``expand_households.repop`` attribute to modify the existing synthetic population

::

  repop:
    steps:
      - input_pre_processor.table_list=repop_input_table_list
      - repop_setup_data_structures
      - initial_seed_balancing.final=true
      - integerize_final_seed_weights.repop
      - repop_balancing
      # expand_households options are append or replace
      - expand_households.repop;append
      - write_results.repop
  
    resume_after: summarize

+----------------+--------------------------------------------------------+
| Attribute      | Description                                            |
+================+========================================================+
| steps          | List of steps to be run |br|                           |
|                | Two options for the expand_households.repop step |br|  |
|                | 1. append |br|                                         |
|                | 2. replace                                             |
+----------------+--------------------------------------------------------+
| resume_after   | The step from which the current run should resume      |
+----------------+--------------------------------------------------------+


**Control Specification File Name for repop mode**:

::

  repop_control_file_name: repop_controls.csv

+---------------------------+--------------------------------------------------------+
| Attribute                 | Description                                            |
+===========================+========================================================+
| repop_control_file_name   | Name of the CSV control specification file for repop   |
+---------------------------+--------------------------------------------------------+


**Input Data Tables for repop mode**

As mentioned earlier, repop mode requires the data pipeline (HDF5 file) from the base run. User should copy the HDF5 file from the base outputs to the repop set up. The data input which needs to be specified in this setting is the control data for the subset of geographies to be modified. Input tables for the repop mode can be specified in the same manner as base mode.

::

  repop_input_table_list:
    - taz_control_data:
      filename : repop_control_totals_taz.csv
      tablename: TAZ_control_data

	  
	  

Specifying Controls
~~~~~~~~~~~~~~~~~~~~~

The controls for a PopulationSim run are specified using the control specification CSV file. Following the ActivitySim framework, Python expressions are used for specifying control constraints.  An example file is below.  

+----------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| target               | geography | seed_table | importance | control_field |  expression                                                         |
+======================+===========+============+============+===============+=====================================================================+
| num_hh               | TAZ       | households | 1000000000 | HHBASE        | (households.WGTP > 0) & (households.WGTP < np.inf)                  |
+----------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_size_4_plus       | TAZ       | households | 5000       | HHSIZE4       | households.NP >= 4                                                  |
+----------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_age_15_24         | TAZ       | households | 500        | HHAGE1        | (households.AGEHOH > 15) & (households.AGEHOH <= 24)                |
+----------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_inc_15            | TAZ       | households | 500        | HHINC1        | (households.HHINCADJ > -999999999) & (households.HHINCADJ <= 21297) |
+----------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| students_fam_housing | TAZ       | persons    | 500        | OSUFAM        | persons.OSUTAG == 1                                                 |
+----------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_wrks_3_plus       | TRACT     | households | 1000       | HHWORK3       | households.NWESR >= 3                                               |
+----------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| hh_by_type_sf        | TRACT     | households | 1000       | SF            | households.HTYPE == 1                                               |
+----------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+
| persons_occ_8        | REGION    | persons    | 1000       | OCCP8         | persons.OCCP == 8                                                   |
+----------------------+-----------+------------+------------+---------------+---------------------------------------------------------------------+

Attribute definitions are as follows:

:target:
        target is the name of the control in PopulationSim
:geography:
        geography is the geographic level of the control, as specified in ``geographies``
:seed_table:
        seed_table is the seed table the control applies to and it can be ``households`` or ``persons``.  If persons, then persons are aggregated to households using the count operator
:importance:
        importance is the importance weight for the control
:control_field:
        control_field is the field in the control data input files that this control applies to
:expression:
        expression is a valid Python/Pandas expression that identifies seed households or persons that this control applies to

  


Error Handling & Debugging
--------------------------

It is recommended to do appropriate checks on input data before running PopulationSim. 

Checks on data inputs
~~~~~~~~~~~~~~~~~~~~~~~

While PopulationSim algorithm is designed to work even with imperfect data, but an error-free and consistent data guarantees optimal performance. Poor performance and errors are usually the result of imperfect data and it is the responsibility of the user to do necessary QA//QC on the input data. Some data problems that are frequently encountered are as follows:

	* Miscoding of data 
	* Inconsistent controls
	* Controls do not add to total number of households
	* Controls do not aggregate consistently
	* missing or mislabelled controls

Common run-time errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a list of common run-time errors:

**Tabs in settings.YAML file**

User should not use /t (tabs) while configuring the settings.YAML file. Presence of /t would result in the following error:

  .. image:: YAML_Tab_Error.JPG
